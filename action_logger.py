import os
import json
from datetime import datetime
class ActionLogger:
    """Logs actions and thoughts for each task"""
    def __init__(self):
        # Create logs directory if it doesn't exist
        self.logs_dir = "action_logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.current_task_file = None
        self.task_count = 0
        
        print(f"Action logs will be saved to: {self.logs_dir}")
    
    def start_new_task(self, command):
        """Start a new task log file"""
        self.task_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_filename = f"task_{self.task_count}_{timestamp}.jsonl"
        self.current_task_file = os.path.join(self.logs_dir, task_filename)
        
        # Create the initial entry
        initial_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "task_start",
            "command": command
        }
        
        # Write the initial entry
        with open(self.current_task_file, "w") as f:
            f.write(json.dumps(initial_entry) + "\n")
            
        print(f"Started new task log: {self.current_task_file}")
        return self.current_task_file
    
    def log_action(self, thought, action, parameters, task_status):
        """Log an action and its corresponding thought"""
        if not self.current_task_file:
            print("Warning: No active task log file")
            return False
            
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "action",
            "thought": thought,
            "action": action,
            "parameters": parameters,
            "task_status": task_status
        }
        
        try:
            with open(self.current_task_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
            return True
        except Exception as e:
            print(f"Error logging action: {e}")
            return False
    
    def log_task_completion(self, success, reason):
        """Log task completion"""
        if not self.current_task_file:
            print("Warning: No active task log file")
            return False
            
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "task_complete",
            "success": success,
            "reason": reason
        }
        
        try:
            with open(self.current_task_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
                
            # Reset current task file since task is complete
            self.current_task_file = None
            return True
        except Exception as e:
            print(f"Error logging task completion: {e}")
            return False
            
    def get_current_task_log(self):
        """Retrieve all entries for the current task
        
        Returns:
            List of log entries for the current task, or empty list if no active task
        """
        if not self.current_task_file or not os.path.exists(self.current_task_file):
            return []
            
        try:
            with open(self.current_task_file, 'r') as f:
                lines = f.readlines()
                
            # Parse all entries
            entries = [json.loads(line) for line in lines]
            return entries
        except Exception as e:
            print(f"Error retrieving current task log: {e}")
            return []
