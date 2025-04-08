import os
import json
from datetime import datetime

class PromptLogger:
    """Logs all prompts sent to the LLM during a program run"""
    def __init__(self):
        # Create logs directory if it doesn't exist
        self.logs_dir = "prompt_logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Create a timestamped file for this program run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.logs_dir, f"prompts_{timestamp}.log")
        
        # Track prompt statistics
        self.prompt_count = 0
        
        # Write header to log file
        with open(self.log_file, "w") as f:
            f.write(f"=== SPOT AGENT PROMPT LOG ===\n")
            f.write(f"Started at: {datetime.now().isoformat()}\n")
        
        print(f"Prompt logs will be saved to: {self.log_file}")
    
    def log_prompt(self, provider, text, prompt):
        """Log a prompt sent to the LLM
        
        Args:
            provider: The LLM provider (ollama, openrouter, google)
            text: The user command or continuation text
            prompt: The full prompt sent to the LLM
        """
        try:
            self.prompt_count += 1
            
            with open(self.log_file, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"PROMPT #{self.prompt_count}\n")
                f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
                f.write(f"PROVIDER: {provider}\n")
                f.write(f"USER COMMAND: {text}\n")
                f.write(f"{'='*80}\n\n")
                f.write(prompt)
                f.write("\n\n")
        except Exception as e:
            print(f"Error logging prompt: {e}")
            
    def log_response(self, response_json):
        """Log the response from the LLM
        
        Args:
            response_json: The JSON response from the LLM
        """
        try:
            with open(self.log_file, "a") as f:
                f.write(f"{'-'*80}\n")
                f.write(f"RESPONSE (at {datetime.now().isoformat()}):\n\n")
                f.write(json.dumps(response_json, indent=2))
                f.write("\n\n")
                
                # Add a summary of the response
                if isinstance(response_json, dict):
                    if "thought" in response_json:
                        f.write(f"Thought: {response_json.get('thought')}\n")
                    if "action" in response_json:
                        f.write(f"Action: {response_json.get('action')}\n")
                    task_status = response_json.get("task_status", {})
                    if task_status:
                        f.write(f"Status: {task_status.get('reason', 'No reason provided')}\n")
                
                f.write(f"{'='*80}\n")
        except Exception as e:
            print(f"Error logging response: {e}")