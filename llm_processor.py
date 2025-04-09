import os
import requests
import json
from google import genai
from pydantic import BaseModel, Field


# LLM System Prompt for Spot control
SPOT_SYSTEM_PROMPT = """
You are an autonomous control system for a Boston Dynamics Spot robot. Your task is to:
1. Interpret human voice commands
2. Reason about the environment and robot state
3. Plan and execute actions to complete tasks
4. Continue reasoning and acting until the task is complete

Environment information available to you:
- odometry logs: Historical data about the robot's position and orientation
- vision logs: Historical data about what the robot has seen

Available actions:
- relative_move(forward_backward, left_right): Move relative to the current position.
- turn(degrees): Turn clockwise (positive) or counterclockwise (negative)
- sit(): Make the robot sit
- stand(): Make the robot stand up
- get_odometry_logs(seconds): Get recent odometry logs for the past seconds
- get_vision_logs(seconds): Get recent vision logs for the past seconds
- task_complete(success, reason): Indicate that the task is complete with success status and reason

First, evaluate the command to understand what you're being asked to do.
Then, retrieve the relevant information about the environment using get_odometry_logs(), get_vision_logs(), or get_terrain_logs(), try to keep the number of seconds to a minimum.
Use this information, if necessary, to plan your actions, execute them, and monitor progress until the task is complete.

Try not to repeat your actions needlessly.

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
    "action": "get_vision_logs",
    "parameters": {"seconds": 1},
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
    forward_backward: float = Field(None, description="Distance in meters to move forward (positive) or backward (negative)")
    left_right: float = Field(None, description="Distance in meters to move right (positive) or left (negative)")
    degrees: float = Field(None, description="Degrees to turn (positive for clockwise, negative for counterclockwise)")
    seconds: int = Field(None, description="Number of seconds to look back for logs")

class SpotCommand(BaseModel):
    """Command for Spot robot"""
    thought: str = Field(..., description="Reasoning about the command")
    action: str = Field(..., description="The action to perform")
    parameters: SpotParameters = Field(..., description="Parameters for the action")
    task_status: TaskStatus = Field(..., description="Current task status")



class LLMProcessor:
    """Processes text through a locally running LLM or cloud API"""
    def __init__(self, prompt_logger=None):
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
    
    def _process_with_ollama(self, text, task_data=None, action_log=None):
        """Process a text command through local LLM with Ollama"""
        try:
            # Format prompt with conversation context
            prompt = f"{SPOT_SYSTEM_PROMPT}\n\n"
            
            # Add action log if available
            if action_log and len(action_log) > 0:
                for i, entry in enumerate(action_log):
                    if entry.get("event_type") == "task_start":
                        prompt += f"Task started with command: {entry.get('command')}\n"
                        prompt += "So far you've done this:\n" if len(action_log) > 1 else "So far no actions have been taken\n"
                    elif entry.get("event_type") == "action":
                        prompt += f"Action {i}: {entry.get('action')}\n"
                        prompt += f"  Thought: {entry.get('thought')}\n"
                        params_str = json.dumps(entry.get('parameters'), indent=2).replace('\n', '\n  ')
                        prompt += f"  Parameters: {params_str}\n"
                        if "task_status" in entry and entry["task_status"].get("reason"):
                            prompt += f"  Status: {entry['task_status'].get('reason')}\n"
                        prompt += "\n"
                prompt += "\n"
            
            # Add current task data if available
            if task_data:
                prompt += f"Current task information:\n{json.dumps(task_data, indent=2)}\n\n"
                
            # Add current command
            prompt += f"User command: {text}"
            
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
            
            # Log the response if logger is available
            if self.prompt_logger and result:
                self.prompt_logger.log_response(result)
                
            return result
                
        except Exception as e:
            print(f"Error processing with Ollama: {e}")
            return None
    
    def _process_with_openrouter(self, text, task_data=None, action_log=None):
        """Process a text command through OpenRouter API"""
        try:
            # Build the messages array with context
            messages = [{"role": "system", "content": SPOT_SYSTEM_PROMPT}]
            
            # Add action log if available
            action_log_text = ""
            if action_log and len(action_log) > 0:
                for i, entry in enumerate(action_log):
                    if entry.get("event_type") == "task_start":
                        action_log_text += f"Task started with command: {entry.get('command')}\n"
                        action_log_text += "So far you've done this:\n" if len(action_log) > 1 else "So far no actions have been taken\n"
                    elif entry.get("event_type") == "action":
                        action_log_text += f"Action {i}: {entry.get('action')}\n"
                        action_log_text += f"  Thought: {entry.get('thought')}\n"
                        params_str = json.dumps(entry.get('parameters'), indent=2).replace('\n', '\n  ')
                        action_log_text += f"  Parameters: {params_str}\n"
                        if "task_status" in entry and entry["task_status"].get("reason"):
                            action_log_text += f"  Status: {entry['task_status'].get('reason')}\n"
                        action_log_text += "\n"
                
                messages.append({"role": "system", "content": action_log_text})
            
            # Add current task data if available
            task_data_text = ""
            if task_data:
                task_data_text = f"Current task information:\n{json.dumps(task_data, indent=2)}"
                messages.append({"role": "system", "content": task_data_text})
                
            # Add current command
            messages.append({"role": "user", "content": text})
            
            # Log the prompt if logger is available
            if self.prompt_logger:
                # Reconstruct the full prompt for logging
                full_prompt = f"{SPOT_SYSTEM_PROMPT}\n\n"
                if action_log_text:
                    full_prompt += action_log_text + "\n"
                if task_data_text:
                    full_prompt += task_data_text + "\n\n"
                full_prompt += f"User command: {text}"
                
                self.prompt_logger.log_prompt("openrouter", text, full_prompt)
            
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
            
            # Log the response if logger is available
            if self.prompt_logger and result:
                self.prompt_logger.log_response(result)
                
            return result
                
        except Exception as e:
            print(f"Error processing with OpenRouter: {e}")
            return None
    
    def _process_with_google(self, text, task_data=None, action_log=None):
        """Process a text command through Google Gemini API using the genai SDK"""
        try:
            # Format prompt with conversation context
            prompt = f"{SPOT_SYSTEM_PROMPT}\n\n"
            
            # Add action log if available
            if action_log and len(action_log) > 0:
                for i, entry in enumerate(action_log):
                    if entry.get("event_type") == "task_start":
                        prompt += f"Task started with command: {entry.get('command')}\n"
                        prompt += "So far you've done this:\n" if len(action_log) > 1 else "So far no actions have been taken\n"
                    elif entry.get("event_type") == "action":
                        prompt += f"Action {i}: {entry.get('action')}\n"
                        prompt += f"  Thought: {entry.get('thought')}\n"
                        params_str = json.dumps(entry.get('parameters'), indent=2).replace('\n', '\n  ')
                        prompt += f"  Parameters: {params_str}\n"
                        if "task_status" in entry and entry["task_status"].get("reason"):
                            prompt += f"  Status: {entry['task_status'].get('reason')}\n"
                        prompt += "\n"
                prompt += "\n"
            
            # Add current task data if available
            if task_data:
                prompt += f"Current task information:\n{json.dumps(task_data, indent=2)}\n\n"
                
            # Add current command
            prompt += f"User command: {text}"
            
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
            
            # Log the response if logger is available
            if self.prompt_logger and result:
                self.prompt_logger.log_response(result)
                
            return result
                
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
