import os
import requests
import json
import time
from google import genai
from pydantic import BaseModel, Field
from utils.timestamp_utils import parse_timestamp
from spot_controller import SpotController


# LLM System Prompt for Spot control
SPOT_SYSTEM_PROMPT = """
You are an autonomous control system for a Boston Dynamics Spot robot. Your task is to:
1. Interpret human voice commands
2. Reason about the environment and robot state
3. Plan and execute actions to complete tasks
4. Continue reasoning and acting until the task is complete
5. Avoid repetitive behaviour, remain vigilant and observant

Environment information available to you:
- odometry logs: Historical data about the robot's position and orientation
- vision logs: Historical data about what the robot has seen
- object detection: Real-time information about objects detected around the robot, including their positions

The system automatically runs object detection in the background. You can see the results in task_data["latest_detected_objects"]. 
Each object has these properties:
- label: The type of object detected (e.g., "person", "chair", "bottle")
- score: The confidence level (0-1) that the object was correctly identified
- position: 3D coordinates relative to the robot's body frame {x, y, z}


Available actions:
- relative_move(x, y): Move relative to the current position (CAN BE USED TO STRAIFE, both x and y can be used together).
- turn(degrees): Turn clockwise (positive) or counterclockwise (negative)
- sit(): Make the robot sit
- stand(): Make the robot stand up
- tilt(pitch, roll, yaw): Make the robot tilt its body with pitch, roll, yaw in radians, 0 being the default for all.
- task_complete(success, reason): Indicate that the task is complete with success status and reason


First, evaluate the command to understand what you're being asked to do.
Use this information, if necessary, to plan your actions, execute them, and monitor progress until the task is complete.

Try not to repeat your actions needlessly.

ALWAYS MAKE SURE YOUR ACTIONS COMPLY WITH YOUR THOUGHTS.

DO NOT REPEAT ACTIONS WITHOUT REASON. CONSIDER IT A DONE TASK
DO NOT NEDLESSLY "MONITOR". IDLING IS MONITORING. THERE IS NO ACTIVE "MONITORING" ACTION. CONSIDER IT A DONE TASK
DO NOT NEDLESSLY "STAND". IDLING IS STANDING. THERE IS NO ACTIVE "STANDING" ACTION. CONSIDER IT A DONE TASK

Respond ONLY with valid JSON in this format:
{
    "thought": "Your reasoning about the current situation",
    "action": "function_name",
    "parameters": {"parameter_name": value},
    "task_status": {
        "complete": true/false,
        "reason": "Explanation of current status"
    }
}

Example response for "move forward 1 meter":
{
    "thought": "I need to move forward 1 meter",
    "action": "relative_move",
    "parameters": {"x": 1},
    "task_status": {
        "complete": true,
        "reason": "Moved forward 1 meter"
    }
}
"""

# Define Pydantic models for structured output
class TaskStatus(BaseModel):
    """Task completion status"""
    complete: bool = Field(..., description="Whether the task is complete")
    reason: str = Field(..., description="Explanation of the current status")

class SpotParameters(BaseModel):
    """Parameters for Spot robot commands"""
    x: float = Field(None, description="Distance in meters to move forward (positive) or backward (negative)")
    y: float = Field(None, description="Distance in meters to move left (positive) or right (negative)")
    degrees: float = Field(None, description="Degrees to turn (positive for clockwise, negative for counterclockwise)")
    seconds: int = Field(None, description="Number of seconds to look back for logs")
    success: bool = Field(None, description="Whether the task was completed successfully")
    reason: str = Field(None, description="Reason for the task completion")
    pitch: float = Field(None, description="Angle in radians to rotate body around y-axis (front down, back up or back down, front up)")
    roll: float = Field(None, description="Angle in radians to rotate body around x-axis (left down, right up or right down, left up)")
    yaw: float = Field(None, description="Angle in radians to rotate body around z-axis (front left, back right of front right, back left)")
    

class SpotCommand(BaseModel):
    """Command for Spot robot"""
    thought: str = Field(..., description="Reasoning about the command")
    action: str = Field(..., description="The action to perform")
    parameters: SpotParameters = Field(..., description="Parameters for the action")
    task_status: TaskStatus = Field(..., description="Current task status")



class LLMProcessor:
    """Processes text through a locally running LLM or cloud API"""
    def __init__(self, prompt_logger=None, spot_controller=None):
        # Determine which LLM provider to use
        self.provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        print(f"Using LLM provider: {self.provider}")
        
        # Initialize Ollama attributes regardless of provider for fallback capability
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "mistral")
        
        # Store conversation context for continued reasoning
        self.conversation_context = []
        
        # Store prompt logger
        self.prompt_logger = prompt_logger
        
        # Store spot controller reference for accessing logs
        self.spot_controller : SpotController = spot_controller
        
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
        except Exception as e:
            print(f"Warning: Could not check Google API model availability: {e}")
            
        print(f"Google API configured with model: {self.google_model}")
    
    def _build_prompt(self, text, task_data=None, action_log=None):
        """Build the prompt with system context, action log, and task data"""
        prompt = f"{SPOT_SYSTEM_PROMPT}\n\n"
        
        # Get the latest vision, odometry, and object detection logs
        #latest_vision_log = self.spot_controller.get_vision_logs(1) if hasattr(self, 'spot_controller') and self.spot_controller else None
        latest_odometry_log = self.spot_controller.get_odometry_logs(1) if hasattr(self, 'spot_controller') and self.spot_controller else None
        #latest_object_detection_log = self.spot_controller.get_object_detection_logs(1) if hasattr(self, 'spot_controller') and self.spot_controller else None
        
        # Add action log if available
        if action_log and len(action_log) > 0:
            for i, entry in enumerate(action_log):
                if entry.get("event_type") == "task_start":
                    timestamp = parse_timestamp(entry.get('timestamp'))
                    time_ago = f" {round(time.time() - timestamp, 1)} seconds ago"
                    prompt += f"Task started with command: {entry.get('command')}{time_ago}.\n"
                    prompt += "So far you've done this:\n" if len(action_log) > 1 else "So far no actions have been taken\n"
                elif entry.get("event_type") == "action":
                    timestamp = parse_timestamp(entry.get('timestamp'))
                    time_ago = f" was taken {round(time.time() - timestamp, 1)} seconds ago"
                    prompt += f"Action {i}: {entry.get('action')}{time_ago}.\n"
                    prompt += f"  Thought: {entry.get('thought')}\n"
                    # Filter out null parameters
                    params = {k:v for k,v in entry.get('parameters', {}).items() if v is not None}
                    if params != {}:
                        params_str = json.dumps(params, indent=2).replace('\n', '\n  ')
                        prompt += f"  Parameters: {params_str}\n"
                    if "task_status" in entry and entry["task_status"].get("reason"):
                        prompt += f"  Status: {entry['task_status'].get('reason')}\n"
                    prompt += "\n"
            prompt += "\n"
        
        # Add current task data including latest perception data if available
        if task_data is None:
            task_data = {}
        
        # Integrate latest logs into task_data
        #if latest_vision_log:
        #    task_data["latest_vision"] = latest_vision_log
        
        if latest_odometry_log:
            task_data["latest_odometry"] = latest_odometry_log
            
        #if latest_object_detection_log and 'objects' in latest_object_detection_log:
        #    # Add the latest detected objects to task_data
        #    task_data["latest_detected_objects"] = latest_object_detection_log['objects']
        
        if task_data:
            prompt += f"Current task information:\n{json.dumps(task_data, indent=2)}\n\n"
        
        # Add current command
        prompt += f"User command: {text}"
        
        return prompt

    def _handle_response(self, provider, text, prompt, result):
        """Common response handling logic"""
        # Log the response if logger is available
        if self.prompt_logger and result:
            self.prompt_logger.log_response(result)
            
        return result

    def _process_with_ollama(self, text, task_data=None, action_log=None):
        """Process a text command through local LLM with Ollama"""
        try:
            prompt = self._build_prompt(text, task_data, action_log)
            
            # Log the prompt if logger is available
            if self.prompt_logger:
                self.prompt_logger.log_prompt("ollama", text, prompt)
            
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
            result = self._extract_json_from_text(response_text)
            
            return self._handle_response("ollama", text, prompt, result)
                
        except Exception as e:
            print(f"Error processing with Ollama: {e}")
            return None
    
    def _process_with_openrouter(self, text, task_data=None, action_log=None):
        """Process a text command through OpenRouter API"""
        try:
            prompt = self._build_prompt(text, task_data, action_log)
            
            # Build messages array for OpenRouter format
            messages = [
                {"role": "system", "content": SPOT_SYSTEM_PROMPT}
            ]
            
            # Add action log and task data as system messages if available
            if action_log and len(action_log) > 0:
                messages.append({"role": "system", "content": prompt[len(SPOT_SYSTEM_PROMPT)+2:prompt.rfind("User command:")]})
            
            # Add current command
            messages.append({"role": "user", "content": text})
            
            # Log the prompt if logger is available
            if self.prompt_logger:
                self.prompt_logger.log_prompt("openrouter", text, prompt)
            
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
            result = self._extract_json_from_text(response_text)
            
            return self._handle_response("openrouter", text, prompt, result)
                
        except Exception as e:
            print(f"Error processing with OpenRouter: {e}")
            return None
    
    def _process_with_google(self, text, task_data=None, action_log=None):
        """Process a text command through Google Gemini API using the genai SDK"""
        try:
            prompt = self._build_prompt(text, task_data, action_log)
            
            # Log the prompt if logger is available
            if self.prompt_logger:
                self.prompt_logger.log_prompt("google", text, prompt)
            
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
            result = None
            if response.parsed is not None:
                # Convert Pydantic object to dictionary
                result = response.parsed.model_dump()
            elif hasattr(response, 'text'):
                # Fallback to text parsing if structured parsing fails
                result = self._extract_json_from_text(response.text)
            else:
                print("Unexpected response format from Google API")
                print(response)
                return None
            
            return self._handle_response("google", text, prompt, result)
                
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
    
    def process_command(self, text, task_data=None, action_log=None):
        """Process a text command through the selected LLM provider"""
        if not text:
            return None
            
        result = None
        
        if self.provider == "ollama":
            result = self._process_with_ollama(text, task_data, action_log)
        elif self.provider == "openrouter":
            result = self._process_with_openrouter(text, task_data, action_log)
            # Fall back to Ollama if OpenRouter fails
            if result is None and os.getenv("OLLAMA_URL"):
                print("OpenRouter failed, falling back to Ollama...")
                result = self._process_with_ollama(text, task_data, action_log)
        elif self.provider == "google":
            result = self._process_with_google(text, task_data, action_log)
            # Fall back to Ollama if Google API fails
            if result is None and os.getenv("OLLAMA_URL"):
                print("Google API failed, falling back to Ollama...")
                result = self._process_with_ollama(text, task_data, action_log)
        else:
            # Fallback to Ollama
            result = self._process_with_ollama(text, task_data, action_log)
        
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
