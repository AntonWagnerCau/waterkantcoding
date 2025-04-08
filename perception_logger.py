import os
import threading
import time
import json
from spot_controller import SpotController
from datetime import datetime

class PerceptionLogger:
    """Runs odometry and vision perception in background and logs data to files"""
    def __init__(self, spot_controller : SpotController):
        self.spot_controller = spot_controller
        self.running = False
        self.odometry_thread = None
        self.vision_thread = None
        self.odometry_interval = float(os.getenv("ODOMETRY_INTERVAL", "0.2"))  # seconds
        self.vision_interval = float(os.getenv("VISION_INTERVAL", "0.5"))  # seconds
        
        # Create logs directory if it doesn't exist
        self.logs_dir = "perception_logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Set up log file paths with ISO timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.odometry_log_path = os.path.join(self.logs_dir, f"odometry_{timestamp}.jsonl")
        self.vision_log_path = os.path.join(self.logs_dir, f"vision_{timestamp}.jsonl")
        
        print(f"Perception logs will be saved to: {self.logs_dir}")
    
    def start(self):
        """Start background perception threads"""
        if self.running:
            print("Perception logging already running")
            return
            
        self.running = True
        
        # Start odometry thread
        self.odometry_thread = threading.Thread(target=self._odometry_loop)
        self.odometry_thread.daemon = True
        self.odometry_thread.start()
        
        # Start vision thread
        self.vision_thread = threading.Thread(target=self._vision_loop)
        self.vision_thread.daemon = True
        self.vision_thread.start()
        
        print("Perception logging started")
    
    def stop(self):
        """Stop background perception threads"""
        if not self.running:
            print("Perception logging not running")
            return
            
        self.running = False
        
        # Wait for threads to terminate
        if self.odometry_thread and self.odometry_thread.is_alive():
            self.odometry_thread.join(timeout=2.0)
        
        if self.vision_thread and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=2.0)
            
        print("Perception logging stopped")
    
    def _log_data(self, file_path, data):
        """Append data entry to log file"""
        try:
            # Add timestamp to data
            timestamped_data = {
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            # Append JSON line to file
            with open(file_path, "a") as f:
                f.write(json.dumps(timestamped_data) + "\n")
                
        except Exception as e:
            print(f"Error logging data: {e}")
    
    def _odometry_loop(self):
        """Background loop for odometry perception"""
        print(f"Odometry logging started, interval: {self.odometry_interval}s")
        
        while self.running:
            try:
                # Get odometry data with caching disabled for background logging
                odometry_data = self.spot_controller.get_odometry()
                
                # Log the data
                self._log_data(self.odometry_log_path, odometry_data)
                
            except Exception as e:
                print(f"Error in odometry loop: {e}")
                
            # Sleep for the configured interval
            time.sleep(self.odometry_interval)
    
    def _vision_loop(self):
        """Background loop for vision perception"""
        print(f"Vision logging started, interval: {self.vision_interval}s")
        
        while self.running:
            try:
                # Get vision data with caching disabled for background logging
                vision_data = self.spot_controller.analyze_images()
                
                # Log the data
                self._log_data(self.vision_log_path, vision_data)
                
            except Exception as e:
                print(f"Error in vision loop: {e}")
                
            # Sleep for the configured interval
            time.sleep(self.vision_interval)
