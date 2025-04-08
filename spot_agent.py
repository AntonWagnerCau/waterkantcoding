"""
SpotAgent - Voice-controlled agent for Boston Dynamics Spot robot
"""
import os
import time
import json
import numpy as np
import pyaudio
import wave
import threading
import queue
import tempfile
import requests
from PIL import Image
import io
import base64
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import bosdyn.client
from bosdyn.client import create_standard_sdk, ResponseError
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client.util import authenticate
from bosdyn.geometry import EulerZXY
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.client.frame_helpers import get_vision_tform_body
from bosdyn.api import image_pb2
from google import genai
from pydantic import BaseModel, Field

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Load environment variables
load_dotenv()

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 500  # Adjusted based on your environment
SILENCE_DURATION = 3  # seconds of silence to stop recording

# LLM System Prompt for Spot control
SPOT_SYSTEM_PROMPT = """
You are an autonomous control system for a Boston Dynamics Spot robot. Your task is to:
1. Interpret human voice commands
2. Reason about the environment and robot state
3. Plan and execute actions to complete tasks
4. Continue reasoning and acting until the task is complete

Environment information available to you:
- odometry: Current position and orientation of the robot
- vision: Description of what the robot sees through its cameras

Available actions:
- relative_move(forward_backward, left_right): Move relative to the current position.
- turn(degrees): Turn clockwise (positive) or counterclockwise (negative)
- sit(): Make the robot sit
- stand(): Make the robot stand up
- take_picture(): Capture an image from the robot's camera
- get_odometry(): Retrieve the robot's current position and orientation
- analyze_image(): Get a description of what the robot currently sees
- task_complete(success, reason): Indicate that the task is complete with success status and reason

First, evaluate the command to understand what you're being asked to do.
Then, gather relevant information about the environment using get_odometry() and analyze_image() ONLY IF YOU NEED TO.
Use analyze_image() only if the user's command requires it.
Use get_odometry() only if the user's command requires it.
Use this information, if necessary, to plan your actions, execute them, and monitor progress until the task is complete.

Many commands can be completed without gathering any information.

Respond ONLY with valid JSON in this format:
{
    "thought": "Your reasoning about the current situation",
    "action": "function_name",
    "parameters": {"parameter_name": value},
    "task_status": {
        "complete": true/false,
        "success": true/false,
        "reason": "Explanation of current status"
    }
}

Example response for "navigate to the chair":
{
    "thought": "I need to find the chair first, then plan a path to it",
    "action": "analyze_image",
    "parameters": {},
    "task_status": {
        "complete": false,
        "success": null,
        "reason": "Need to locate the chair first"
    }
}
"""

# Define Pydantic models for structured output
class TaskStatus(BaseModel):
    """Task completion status"""
    complete: bool = Field(..., description="Whether the task is complete")
    success: bool = Field(..., description="Whether the task was completed successfully")
    reason: str = Field(..., description="Explanation of the current status")

class SpotParameters(BaseModel):
    """Parameters for Spot robot commands"""
    forward_backward: float = Field(..., description="Distance in meters to move forward (positive) or backward (negative)")
    left_right: float = Field(..., description="Distance in meters to move right (positive) or left (negative)")
    degrees: float = Field(..., description="Degrees to turn (positive for clockwise, negative for counterclockwise)")

class SpotCommand(BaseModel):
    """Command for Spot robot"""
    thought: str = Field(..., description="Reasoning about the command")
    action: str = Field(..., description="The action to perform")
    parameters: SpotParameters = Field(..., description="Parameters for the action")
    task_status: TaskStatus = Field(..., description="Current task status")

class AudioProcessor:
    """Handles microphone input and speech-to-text conversion using offline Whisper"""
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # Initialize Whisper model
        # Options: "base", "small", "medium", "large"
        model_size = os.getenv("WHISPER_MODEL", "base.en")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float32")
        print(f"Loading Whisper model '{model_size}' (this may take a moment)...")
        
        # Use CPU by default, but you can set device="cuda" if you have a GPU
        self.model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        print("Whisper model loaded successfully")
        
    def start_recording(self):
        """Start recording audio from microphone"""
        self.is_recording = True
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print("Listening... (speak now)")
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
    def _record_audio(self):
        """Record audio until silence is detected"""
        frames = []
        silent_chunks = 0
        silent_threshold = int(SILENCE_DURATION * RATE / CHUNK)
        
        while self.is_recording:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            # Detect silence
            audio_data = np.frombuffer(data, dtype=np.int16)
            if np.abs(audio_data).mean() < SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks > silent_threshold:
                    break
            else:
                silent_chunks = 0
        
        # Save the recorded audio for Whisper processing
        if len(frames) > 0:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_filename = temp_audio.name
                
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            self.audio_queue.put(temp_filename)
        
        # Close the stream
        self.stream.stop_stream()
        self.stream.close()
        self.is_recording = False
        print("Finished recording")
        
    def get_transcription(self):
        """Convert recorded audio to text using Whisper"""
        if self.audio_queue.empty():
            return None
            
        audio_file = self.audio_queue.get()
        print("Transcribing audio...")
        
        try:
            # Transcribe with Whisper
            segments, _ = self.model.transcribe(audio_file, beam_size=5)
            
            # Combine all segments into one text
            text = " ".join([segment.text for segment in segments])
            
            print(f"You said: {text}")
            return text
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None
        finally:
            # Clean up the temporary file
            if os.path.exists(audio_file):
                os.remove(audio_file)

class SpotController:
    """Controls the Boston Dynamics Spot robot"""
    def __init__(self):
        self.connected = False
        self.robot = None
        self.command_client = None
        self.state_client = None
        self.image_client = None
        self.last_image_path = None
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.orientation = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        
        # Initialize vision model
        self.vision_model = None
        self.vision_processor = None
        self.setup_vision_model()
        
    def setup_vision_model(self):
        """Initialize image captioning model"""
        try:
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading vision model on {device}...")
            
            # Use the smaller and more reliable BLIP model
            self.vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
            
            print("Vision model loaded successfully")
        except Exception as e:
            print(f"Error setting up vision model: {e}")
            print("Will use basic image processing instead")
        
    def connect(self):
        """Connect to the Spot robot"""
        try:
            # Initialize the SDK
            sdk = create_standard_sdk("SpotAgentClient")
            self.robot = sdk.create_robot(os.getenv("SPOT_IP"))
            
            self.robot.authenticate(os.getenv("SPOT_USERNAME"), os.getenv("SPOT_PASSWORD"))
            
            # Sync with the robot's time
            self.robot.time_sync.wait_for_sync()
            
            # Acquire lease
            self.lease_client : LeaseClient = self.robot.ensure_client('lease')
            self.lease = self.lease_client.take()
            
            # Get command clients
            self.command_client : RobotCommandClient = self.robot.ensure_client(RobotCommandClient.default_service_name)
            self.state_client : RobotStateClient = self.robot.ensure_client(RobotStateClient.default_service_name)
            self.image_client : ImageClient = self.robot.ensure_client(ImageClient.default_service_name)
            
            # Power on the robot
            self.robot.power_on(timeout_sec=20)
            assert self.robot.is_powered_on(), "Robot power on failed"
            
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to Spot: {e}")
            return False
    
    def get_odometry(self):
        """Retrieve the robot's current position and orientation"""
        if not self.connected:
            print("Robot not connected, returning simulated odometry data")
            # Return simulated data in simulation mode
            return {
                "position": self.position,
                "orientation": self.orientation
            }
        
        try:
            # Get robot state
            robot_state = self.state_client.get_robot_state()
            
            state = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)
            position = state.position
            self.position = {
                "x": position.x, 
                "y": position.y, 
                "z": position.z
            }
            
            rotation = state.rotation
            self.orientation = {
                "roll": rotation.to_roll(),
                "pitch": rotation.to_pitch(),
                "yaw": rotation.to_yaw()
            }
            
            return {
                "position": self.position,
                "orientation": self.orientation
            }
        except Exception as e:
            print(f"Error getting odometry: {e}")
            # Return the last known position in case of error
            return {
                "position": self.position,
                "orientation": self.orientation
            }
    
    def analyze_image(self):
        """Capture an image and generate a caption of what the robot sees"""
        # First capture an image
        image_path = self.take_picture()
        
        if not image_path:
            print("Failed to capture image")
            return {"description": "Unable to capture image"}
        
        # If we don't have a vision model, return a placeholder
        if self.vision_model is None:
            print("No vision model available: returning placeholder image description")
            return {"description": "Simulated image analysis: The robot appears to be in a room with various objects."}
        
        # Use local vision model to analyze the image
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Unconditional image captioning
            inputs = self.vision_processor(image, return_tensors="pt").to(self.vision_model.device)
            
            # Generate caption
            out = self.vision_model.generate(**inputs, max_length=30)
            caption = self.vision_processor.decode(out[0], skip_special_tokens=True)
            
            # Create description
            description = f"The robot sees: {caption}"
            
            # Try to extract objects for better reasoning
            try:
                import re
                # Extract potential objects from the caption
                nouns = re.findall(r'\b[a-z]+\b', caption.lower())
                common_objects = ["person", "chair", "table", "door", "wall", "floor", "window", 
                                 "book", "box", "laptop", "computer", "desk", "room", "office", 
                                 "kitchen", "bed", "couch", "sofa"]
                detected_objects = [noun for noun in nouns if noun in common_objects]
                
                if detected_objects:
                    description += f" Objects detected: {', '.join(detected_objects)}."
            except:
                # If additional analysis fails, just use the caption
                pass
            
            return {"description": description}
            
        except Exception as e:
            print(f"Error analyzing image with vision model: {e}")
            return {"description": f"Image analysis failed due to an error: {str(e)}"}
    
    def relative_move(self, delta_x, delta_y):
        """Command the robot to move relative to the current position"""
        if not self.connected:
            print("Robot not connected, simulating movement")
            # Update simulated position
            self.position["x"] += delta_x * np.cos(np.radians(self.orientation["yaw"]))
            self.position["y"] += delta_x * np.sin(np.radians(self.orientation["yaw"]))
            self.position["x"] += -delta_y * np.sin(np.radians(self.orientation["yaw"]))
            self.position["y"] += delta_y * np.cos(np.radians(self.orientation["yaw"]))
            return True
        
        try:
            cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                goal_x_rt_body=delta_x,
                goal_y_rt_body=delta_y,
                goal_heading_rt_body=0,
                frame_tree_snapshot=self.robot.get_frame_tree_snapshot()
            )   
            self.command_client.robot_command(cmd, end_time_secs=time.time() + 5)
            return True
        except Exception as e:
            print(f"Error relative move: {e}")
            return False
            
    def walk_forward(self, distance_meters):
        """Command the robot to walk forward"""
        return self.relative_move(distance_meters, 0)
    
    def walk_backward(self, distance_meters):
        """Command the robot to walk backward"""
        return self.relative_move(-distance_meters, 0)
    
    def turn(self, degrees):
        """Command the robot to turn by specified degrees"""
        if not self.connected:
            print("Robot not connected, simulating turn")
            # Update simulated orientation
            self.orientation["yaw"] = (self.orientation["yaw"] + degrees) % 360
            return True
            
        try:
            # Convert degrees to radians
            radians = np.radians(-degrees)
            
            # Determine turn direction and angular velocity
            angular_velocity = 0.5 if degrees > 0 else -0.5  # rad/s
            
            # Calculate duration
            duration = abs(radians / angular_velocity)
            
            cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                goal_x_rt_body=0,
                goal_y_rt_body=0,
                goal_heading_rt_body=radians,
                frame_tree_snapshot=self.robot.get_frame_tree_snapshot()
            )
            self.command_client.robot_command(cmd, end_time_secs=time.time() + duration)
            return True
        except Exception as e:
            print(f"Error turning: {e}")
            return False
    
    def sit(self):
        """Command the robot to sit"""
        if not self.connected:
            print("Robot not connected, simulating sit")
            return True
            
        try:
            cmd = RobotCommandBuilder.synchro_sit_command()
            self.command_client.robot_command(cmd, end_time_secs=time.time() + 1)
            return True
        except Exception as e:
            print(f"Error sitting: {e}")
            return False
    
    def stand(self):
        """Command the robot to stand"""
        if not self.connected:
            print("Robot not connected, simulating stand")
            return True
            
        try:
            cmd = RobotCommandBuilder.synchro_stand_command()
            self.command_client.robot_command(cmd, end_time_secs=time.time() + 1)
            return True
        except Exception as e:
            print(f"Error standing: {e}")
            return False

    def take_picture(self):
        """Capture an image from the robot's front camera"""
        if not self.connected:
            print("Robot not connected, returning simulated image path")
            # In simulation mode, use a placeholder timestamp
            self.last_image_path = f"spot_image_sim_{int(time.time())}.jpg"
            
            # Create a simple test image
            img = Image.new('RGB', (640, 480), color=(73, 109, 137))
            img.save(self.last_image_path)
            
            print(f"Simulated image saved as {self.last_image_path}")
            return self.last_image_path
        
        try:
            # Get image client
            image_client = self.robot.ensure_client('image')
            
            # Request image from front camera
            image_response = image_client.get_image_from_sources(['frontleft_fisheye_image'])
            
            # Save the image
            if len(image_response) > 0:
                self.last_image_path = f"spot_image_{int(time.time())}.jpg"
                with open(self.last_image_path, 'wb') as outfile:
                    outfile.write(image_response[0].shot.image.data)
                print(f"Image saved as {self.last_image_path}")
                return self.last_image_path
            else:
                print("No image data received")
                return None
        except Exception as e:
            print(f"Error taking picture: {e}")
            return None
    
    def disconnect(self):
        """Disconnect from the robot"""
        if self.connected:
            try:
                # Return the lease
                self.lease_client.return_lease(self.lease)
                
                # Power off robot
                self.robot.power_off(cut_immediately=False)
                
                self.connected = False
                print("Disconnected from Spot")
            except Exception as e:
                print(f"Error disconnecting: {e}")

class LLMProcessor:
    """Processes text through a locally running LLM or cloud API"""
    def __init__(self):
        # Determine which LLM provider to use
        self.provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        print(f"Using LLM provider: {self.provider}")
        
        # Initialize Ollama attributes regardless of provider for fallback capability
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "mistral")
        
        # Store conversation context for continued reasoning
        self.conversation_context = []
        
        if self.provider == "ollama":
            self._setup_ollama()
        elif self.provider == "openrouter":
            self._setup_openrouter()
        elif self.provider == "google":
            self._setup_google()
        else:
            print(f"Warning: Unknown LLM provider '{self.provider}', defaulting to Ollama")
            self.provider = "ollama"
            self._setup_ollama()
    
    def _setup_ollama(self):
        """Setup for Ollama LLM provider"""
        # Check if Ollama is running
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code != 200:
                print("Warning: Ollama server not responding properly")
            else:
                print("Ollama server detected")
                models = response.json()
                print(f"Available models: {', '.join([model['name'] for model in models.get('models', [])])}")
        except requests.exceptions.ConnectionError:
            print("Warning: Ollama server not running. Please start it with 'ollama serve'")
            print("You can install Ollama from https://ollama.com/")
            print("Then pull a model with 'ollama pull mistral' or another model of your choice")
    
    def _setup_openrouter(self):
        """Setup for OpenRouter LLM provider"""
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "google/gemini-1.5-pro-latest")
        
        if not self.openrouter_api_key:
            print("Warning: OpenRouter API key not set. Please set OPENROUTER_API_KEY in your .env file")
            return
            
        print(f"OpenRouter configured with model: {self.openrouter_model}")
    
    def _setup_google(self):
        """Setup for Google Gemini API provider"""
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_model = os.getenv("GOOGLE_MODEL", "gemini-1.5-pro-latest")
        
        if not self.google_api_key:
            print("Warning: Google API key not set. Please set GOOGLE_API_KEY in your .env file")
            return
        
        # Configure the Google client
        self.google_client = genai.Client(api_key=self.google_api_key)
        
        # Check model availability
        try:
            # List available models
            models = self.google_client.models.list()
            model_names = [model.name for model in models]
            print(f"Available Google models: {', '.join(model_names)}")
        except Exception as e:
            print(f"Warning: Could not check Google API model availability: {e}")
            
        print(f"Google API configured with model: {self.google_model}")
    
    def _process_with_ollama(self, text, task_data=None):
        """Process a text command through local LLM with Ollama"""
        try:
            # Format prompt with conversation context
            prompt = f"{SPOT_SYSTEM_PROMPT}\n\n"
            
            # Add conversation context if available
            if self.conversation_context:
                prompt += "Previous interactions:\n"
                for i, (msg, response) in enumerate(self.conversation_context):
                    prompt += f"Command {i+1}: {msg}\n"
                    prompt += f"Response {i+1}: {response}\n"
                prompt += "\n"
            
            # Add current task data if available
            if task_data:
                prompt += f"Current task information:\n{json.dumps(task_data, indent=2)}\n\n"
                
            # Add current command
            prompt += f"User command: {text}"
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                print(f"Error from Ollama: {response.text}")
                return None
                
            response_json = response.json()
            response_text = response_json.get("response", "")
            
            # Extract JSON from response
            return self._extract_json_from_text(response_text)
                
        except Exception as e:
            print(f"Error processing with Ollama: {e}")
            return None
    
    def _process_with_openrouter(self, text, task_data=None):
        """Process a text command through OpenRouter API"""
        try:
            # Build the messages array with context
            messages = [{"role": "system", "content": SPOT_SYSTEM_PROMPT}]
            
            # Add conversation context if available
            for msg, response in self.conversation_context:
                messages.append({"role": "user", "content": msg})
                messages.append({"role": "assistant", "content": response})
            
            # Add current task data if available
            if task_data:
                task_prompt = f"Current task information:\n{json.dumps(task_data, indent=2)}"
                messages.append({"role": "system", "content": task_prompt})
                
            # Add current command
            messages.append({"role": "user", "content": text})
            
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.openrouter_model,
                    "messages": messages
                }
            )
            
            if response.status_code != 200:
                print(f"Error from OpenRouter: {response.text}")
                return None
                
            response_json = response.json()
            response_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Extract JSON from response
            return self._extract_json_from_text(response_text)
                
        except Exception as e:
            print(f"Error processing with OpenRouter: {e}")
            return None
    
    def _process_with_google(self, text, task_data=None):
        """Process a text command through Google Gemini API using the genai SDK"""
        try:
            # Format prompt with conversation context
            prompt = f"{SPOT_SYSTEM_PROMPT}\n\n"
            
            # Add conversation context if available
            if self.conversation_context:
                prompt += "Previous interactions:\n"
                for i, (msg, response) in enumerate(self.conversation_context):
                    prompt += f"Command {i+1}: {msg}\n"
                    prompt += f"Response {i+1}: {response}\n"
                prompt += "\n"
            
            # Add current task data if available
            if task_data:
                prompt += f"Current task information:\n{json.dumps(task_data, indent=2)}\n\n"
                
            # Add current command
            prompt += f"User command: {text}"
            
            # Generate content with JSON format using the Client approach with Pydantic schema
            response = self.google_client.models.generate_content(
                model=self.google_model,
                contents=prompt,
                config={
                    'temperature': 0.2,
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 1024,
                    'response_mime_type': 'application/json',
                    'response_schema': SpotCommand,
                }
            )
            
            # Check if we have a parsed response (Pydantic model)
            if response.parsed is not None:
                # Convert Pydantic object to dictionary
                return response.parsed.model_dump()
            elif hasattr(response, 'text'):
                # Fallback to text parsing if structured parsing fails
                return self._extract_json_from_text(response.text)
            else:
                print("Unexpected response format from Google API")
                print(response)
                return None
                
        except Exception as e:
            print(f"Error processing with Google API: {e}")
            return None
    
    def _extract_json_from_text(self, text):
        """Extract JSON objects from text response"""
        try:
            # Try to extract JSON from the response
            # Look for JSON pattern in the response
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_text = text[json_start:json_end]
                action_data = json.loads(json_text)
                return action_data
            else:
                print("No valid JSON found in LLM response")
                print(f"LLM response: {text}")
                return None
        except json.JSONDecodeError:
            print(f"Error parsing LLM response: {text}")
            return None
    
    def process_command(self, text, task_data=None):
        """Process a text command through the selected LLM provider"""
        if not text:
            return None
            
        result = None
        
        if self.provider == "ollama":
            result = self._process_with_ollama(text, task_data)
        elif self.provider == "openrouter":
            result = self._process_with_openrouter(text, task_data)
            # Fall back to Ollama if OpenRouter fails
            if result is None and os.getenv("OLLAMA_URL"):
                print("OpenRouter failed, falling back to Ollama...")
                result = self._process_with_ollama(text, task_data)
        elif self.provider == "google":
            result = self._process_with_google(text, task_data)
            # Fall back to Ollama if Google API fails
            if result is None and os.getenv("OLLAMA_URL"):
                print("Google API failed, falling back to Ollama...")
                result = self._process_with_ollama(text, task_data)
        else:
            # Fallback to Ollama
            result = self._process_with_ollama(text, task_data)
        
        # Update conversation context if we got a result
        if result:
            # Convert the result to a string for storage
            result_str = json.dumps(result)
            self.conversation_context.append((text, result_str))
            
            # Keep context to a reasonable size
            if len(self.conversation_context) > 5:
                self.conversation_context.pop(0)
                
        return result
        
    def reset_conversation(self):
        """Reset the conversation context"""
        self.conversation_context = []
        print("Conversation context reset")

def main():
    """Main application function"""
    print("Starting SpotAgent...")
    audio_processor = AudioProcessor()
    llm_processor = LLMProcessor()
    spot_controller = SpotController()
    
    # Connect to Spot robot
    print("Connecting to Spot robot...")
    if os.getenv("SPOT_IP"):
        spot_connected = spot_controller.connect()
        if spot_connected:
            print("Connected to Spot successfully!")
        else:
            print("Failed to connect to Spot. Running in simulation mode.")
    else:
        print("No Spot robot configuration found. Running in simulation mode.")
        spot_connected = False
    
    # Main loop
    try:
        while True:
            # Start recording for initial command
            audio_processor.start_recording()
            
            # Wait for recording to finish
            while audio_processor.is_recording:
                time.sleep(0.1)
            
            # Get transcription
            text = audio_processor.get_transcription()
            
            # Exit if asked
            if text and any(keyword in text.lower() for keyword in ["exit", "quit", "stop"]):
                print("Exiting application...")
                break
            
            # Process with LLM and start task execution loop
            if text:
                # Reset conversation for new task
                llm_processor.reset_conversation()
                
                # Initialize task state
                task_complete = False
                task_data = None
                
                # Task execution loop - continue until task is marked complete
                while not task_complete:
                    # Process current state with LLM
                    action_data = llm_processor.process_command(text if task_data is None else "Continue task execution", task_data)
                    
                    if not action_data:
                        print("Failed to get a valid response from LLM")
                        break
                    
                    # Extract task status
                    task_status = action_data.get('task_status', {})
                    task_complete = task_status.get('complete', False) or task_status.get('success', False) or action_data["action"] is None or action_data.get('action', '').lower() in ['', 'none', 'stop', 'exit', 'quit', 'null']
                    
                    print(f"Thought: {action_data.get('thought')}")
                    print(f"Action: {action_data.get('action')}")
                    print(f"Parameters: {action_data.get('parameters')}")
                    print(f"Task status: {task_status}")
                    
                    # Execute command based on action
                    action = action_data.get('action')
                    params = action_data.get('parameters', {})
                    
                    # Store results for next iteration
                    task_data = {"last_action": action, "last_result": None}
                    
                    # Execute the action
                    if action == 'walk_forward' or action == 'relative_move' and params.get('forward_backward', 0) > 0:
                        result = spot_controller.walk_forward(params.get('forward_backward', 1.0))
                        task_data["last_result"] = {"success": result}
                    
                    elif action == 'walk_backward' or action == 'relative_move' and params.get('forward_backward', 0) < 0:
                        result = spot_controller.walk_backward(abs(params.get('forward_backward', 1.0)))
                        task_data["last_result"] = {"success": result}
                    
                    elif action == 'relative_move':
                        result = spot_controller.relative_move(
                            params.get('forward_backward', 0), 
                            params.get('left_right', 0)
                        )
                        task_data["last_result"] = {"success": result}
                    
                    elif action == 'turn':
                        result = spot_controller.turn(params.get('degrees', 90))
                        task_data["last_result"] = {"success": result}
                    
                    elif action == 'sit':
                        result = spot_controller.sit()
                        task_data["last_result"] = {"success": result}
                    
                    elif action == 'stand':
                        result = spot_controller.stand()
                        task_data["last_result"] = {"success": result}
                    
                    elif action == 'take_picture':
                        image_path = spot_controller.take_picture()
                        task_data["last_result"] = {"success": image_path is not None, "image_path": image_path}
                    
                    elif action == 'get_odometry':
                        odometry = spot_controller.get_odometry()
                        task_data["last_result"] = odometry
                    
                    elif action == 'analyze_image':
                        vision_result = spot_controller.analyze_image()
                        task_data["last_result"] = vision_result
                    
                    elif action == 'task_complete':
                        # The LLM has decided the task is complete
                        task_complete = True
                        success = params.get('success', True)
                        reason = params.get('reason', "Task completed successfully")
                        print(f"Task completed: {success}, Reason: {reason}")
                    
                    else:
                        print(f"Unknown action: {action}")
                        task_data["last_result"] = {"error": f"Unknown action: {action}"}
                    
                    # Pause briefly between actions
                    time.sleep(1)
                
                print("\nTask execution complete. Ready for next command.")
            
            print("\nReady for next command. Speak now...")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        if spot_connected:
            spot_controller.disconnect()
        print("SpotAgent terminated")

if __name__ == "__main__":
    main() 