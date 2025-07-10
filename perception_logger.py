import os
import threading
import time
import json
import numpy as np
from spot_controller import SpotController
from datetime import datetime
import io
import base64
from PIL import Image # Make sure PIL is imported

class PerceptionLogger:
    """Runs odometry and vision perception in background and logs data to files"""
    def __init__(self, spot_controller : SpotController, state_update_callback=None):
        self.spot_controller = spot_controller
        self.state_update_callback = state_update_callback  # Add callback for state updates
        self.running = False
        self.odometry_thread = None
        self.vision_thread = None # Keep for legacy vision logging if needed
        self.terrain_thread = None
        self.object_detection_thread = None  # New thread for object detection
        self.odometry_interval = float(os.getenv("ODOMETRY_INTERVAL", "0"))
        self.vision_interval = float(os.getenv("VISION_INTERVAL", "0"))
        self.terrain_interval = float(os.getenv("TERRAIN_INTERVAL", "0"))
        self.object_detection_interval = float(os.getenv("OBJECT_DETECTION_INTERVAL", "0"))
        # Placeholder for person direction thread & interval; will initialize after timestamp
        
        # Create logs directory if it doesn't exist
        self.logs_dir = "perception_logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Set up log file paths with ISO timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.odometry_log_path = os.path.join(self.logs_dir, f"odometry_{timestamp}.jsonl")
        self.vision_log_path = os.path.join(self.logs_dir, f"vision_{timestamp}.jsonl") # For legacy
        self.terrain_log_path = os.path.join(self.logs_dir, f"terrain_{timestamp}.jsonl")
        self.object_detection_log_path = os.path.join(self.logs_dir, f"objects_{timestamp}.jsonl")

        # --- Person direction (unit vectors to persons) ---
        self.person_interval = float(os.getenv("PERSON_VECTOR_INTERVAL", "0.2"))  # seconds between runs
        self.person_thread = None
        self.person_vectors_log_path = os.path.join(self.logs_dir, f"person_vectors_{timestamp}.jsonl")
        
        print(f"Perception logs will be saved to: {self.logs_dir}")
        # print(f"Annotated images will be saved to: {self.annotated_image_dir}") # Removed
    
    def start(self):
        """Start background perception threads"""
        if self.running:
            print("Perception logging already running")
            return
            
        self.running = True
        
        # Start odometry thread
        #self.odometry_thread = threading.Thread(target=self._odometry_loop)
        #self.odometry_thread.daemon = True
        #self.odometry_thread.start()
        
        # Start vision thread
        self.vision_thread = threading.Thread(target=self._vision_loop)
        self.vision_thread.daemon = True
        #self.vision_thread.start()
        
        # Start terrain thread
        #self.terrain_thread = threading.Thread(target=self._terrain_loop)
        #self.terrain_thread.daemon = True
        #self.terrain_thread.start()
        
        # Start object detection thread
        self.object_detection_thread = threading.Thread(target=self._object_detection_loop)
        self.object_detection_thread.daemon = True
        #self.object_detection_thread.start()

        # Start person direction thread
        self.person_thread = threading.Thread(target=self._person_direction_loop)
        self.person_thread.daemon = True
        self.person_thread.start()
        
        print("Perception logging started")
    
    def stop(self):
        """Stop background perception threads"""
        if not self.running:
            # print("Perception logging not running") # Reduce noise
            return
            
        print("Stopping perception logging...")
        self.running = False
        threads_to_join = [
            self.odometry_thread, 
            self.vision_thread, 
            self.terrain_thread, 
            self.object_detection_thread
            , self.person_thread
        ]
        
        for thread in threads_to_join:
             if thread and thread.is_alive():
                 thread.join(timeout=2.0)
            
        print("Perception logging stopped")
    
    def _log_data(self, file_path, data):
        """Append data entry to log file"""
        try:
            # Add timestamp to data
            timestamped_data = {
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            # Need custom serializer for non-serializable data if any (e.g., numpy arrays if not converted)
            def default_serializer(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                # Add other non-serializable types here if needed
                raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

            # Append JSON line to file
            with open(file_path, "a") as f:
                # Use default serializer to handle potential numpy arrays etc.
                f.write(json.dumps(timestamped_data, default=default_serializer) + "\n")
                
        except Exception as e:
            # Avoid spamming logs if file logging itself fails repeatedly
            # Consider adding a counter or rate limiting here
            print(f"Error logging data to {os.path.basename(file_path)}: {e}")
    
    def _odometry_loop(self):
        """Background loop for odometry perception"""
        while self.running:
            start_time = time.time()
            try:
                odometry_data = self.spot_controller.get_odometry()
                if odometry_data:
                    self._log_data(self.odometry_log_path, odometry_data)
                    # Send update via callback if available
                    if self.state_update_callback:
                        self.state_update_callback({"odometry": odometry_data})
            except Exception as e:
                print(f"Error in odometry loop: {e}")
            
            # Sleep accounting for processing time
            elapsed = time.time() - start_time
            sleep_time = max(0, self.odometry_interval - elapsed)
            if self.running: # Check again before sleeping
                time.sleep(sleep_time)
    
    def _vision_loop(self):
        """Background loop for legacy vision perception"""
        while self.running:
            start_time = time.time()
            try:
                # Get vision data with caching disabled for background logging
                vision_data = self.spot_controller.analyze_images()
                
                # Log the data
                self._log_data(self.vision_log_path, vision_data)
                
            except Exception as e:
                print(f"Error in vision loop: {e}")
                
            # Sleep accounting for processing time
            elapsed = time.time() - start_time
            sleep_time = max(0, self.vision_interval - elapsed)
            if self.running: # Check again before sleeping
                time.sleep(sleep_time)
    
    def _object_detection_loop(self):
        """Background loop for object detection with Base64 image transfer and camera details."""
        while self.running:
            start_time = time.time()
            detection_data = None # Define outside try block
            try:
                if self.spot_controller.connected:
                    # Request images, object locations, and camera details
                    results = self.spot_controller.locate_objects_in_view(return_images=True)
                    
                    if results and 'error' not in results:
                        objects = results.get('objects', [])
                        images_data = results.get('images', {})
                        camera_details = results.get('camera_details', {}) # Get camera details
                        error = None
                        
                        # Process images: Encode to Base64
                        base64_images = {}
                        for camera_name, img_info in images_data.items():
                            annotated_image = img_info.get('annotated_image')
                            if annotated_image and isinstance(annotated_image, Image.Image):
                                try:
                                    buffer = io.BytesIO()
                                    # Save PIL Image to buffer as JPEG
                                    annotated_image.save(buffer, format="JPEG", quality=100) # Adjust quality as needed
                                    buffer.seek(0)
                                    img_bytes = buffer.read()
                                    # Encode bytes to Base64 string
                                    base64_str = base64.b64encode(img_bytes).decode('utf-8')
                                    base64_images[camera_name] = base64_str
                                except Exception as encode_err:
                                    print(f"Error encoding image {camera_name}: {encode_err}") # Keep actual errors
                            else:
                                # Keep warning for missing images
                                print(f"Warning: No valid annotated image found for {camera_name}")
                        
                        # Prepare data payload for logging and WebSocket
                        detection_data = {
                            "status": "success",
                            "objects": objects,
                            "object_count": len(objects),
                            "base64_images": base64_images, # Send Base64 images
                            "camera_details": camera_details # Send camera details
                        }
                        
                    else: # Handle error from locate_objects_in_view
                        error = results.get('error') if results else "Unknown error locating objects"
                        print(f"Object detection failed: {error}") # Keep actual errors
                        detection_data = {"status": "error", "error": str(error), "camera_details": results.get('camera_details', {})}
                
                else: # Spot not connected
                    # print("[_object_detection_loop] Spot not connected") # REMOVED DEBUG
                    detection_data = {"status": "error", "error": "Spot not connected", "camera_details": {}}

                # Send update via callback (if data was prepared)
                if detection_data and self.state_update_callback:
                    # print("[_object_detection_loop] Sending object_detection update via callback.") # REMOVED DEBUG
                    self.state_update_callback({"object_detection": detection_data})
                # elif not self.state_update_callback:
                #      print("[_object_detection_loop] State update callback not available.") # REMOVED DEBUG
            
                # Log the data (even if error state)
                if detection_data:
                     # Summarize base64 data and camera details for logs
                     log_payload = detection_data.copy()
                     if "base64_images" in log_payload and isinstance(log_payload["base64_images"], dict):
                          log_payload["base64_images"] = {cam: f"<base64_len_{len(data)}...>" for cam, data in log_payload["base64_images"].items()}
                     if "camera_details" in log_payload and isinstance(log_payload["camera_details"], dict):
                          log_payload["camera_details"] = {cam: "<details...>" for cam in log_payload["camera_details"]}
                     self._log_data(self.object_detection_log_path, log_payload)

            except Exception as e:
                # Keep critical error logging
                print(f"Critical Error in object detection loop: {e}")
                error_payload = {"status": "error", "error": f"Loop error: {str(e)}", "camera_details": {}}
                self._log_data(self.object_detection_log_path, error_payload)
                if self.state_update_callback:
                     self.state_update_callback({"object_detection": error_payload})

            # Sleep accounting for processing time
            elapsed = time.time() - start_time
            sleep_time = max(0, self.object_detection_interval - elapsed)
            if self.running:
                time.sleep(sleep_time)

    def _person_direction_loop(self):
        """Background loop that computes unit direction vectors to detected persons."""
        while self.running:
            start_time = time.time()
            try:
                vectors = []
                if self.spot_controller.connected:
                    vectors = self.spot_controller.get_person_direction_vectors(threshold=0.25)
                    
                # Update SpotController attribute for consumption by LLMProcessor
                self.spot_controller.latest_person_vectors = vectors

                # Log to file
                self._log_data(self.person_vectors_log_path, {"person_vectors": vectors})

                # Broadcast via callback
                if self.state_update_callback:
                    self.state_update_callback({"person_vectors": vectors})

            except Exception as e:
                print(f"Error in person direction loop: {e}")

            # Sleep until next run
            elapsed = time.time() - start_time
            sleep_time = max(0, self.person_interval - elapsed)
            if self.running:
                time.sleep(sleep_time)

    def analyze_terrain(self):
        """Analyze terrain height grid data around the robot
        
        Processes the 5x5 meter area around the robot with 3x3cm grid resolution
        
        Returns:
            Dictionary with terrain analysis results
        """
        try:
            # Get current robot position
            odometry = self.spot_controller.get_odometry()
            robot_position = odometry["position"]
            
            # Assuming a method get_terrain_grid exists or will be implemented
            # This would return a grid of height values centered on the robot
            height_grid = self._get_terrain_grid()
            
            if height_grid is None:
                return {
                    "status": "error",
                    "message": "Could not retrieve terrain height grid"
                }
            
            # Analyze the terrain
            analysis = self._analyze_height_grid(height_grid)
            
            # Return combined data
            return {
                "robot_position": robot_position,
                "grid_resolution_cm": 3,
                "grid_size_m": 5,
                "analysis": analysis
            }
            
        except Exception as e:
            print(f"Error analyzing terrain: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _get_terrain_grid(self):
        """Get terrain height grid from the robot
        
        Returns:
            2D numpy array of height values, or None if not available
        """
        try:
            # Call the new get_terrain_grid method in SpotController
            # Default parameters: 5x5 meter grid with 3cm resolution
            height_grid = self.spot_controller.get_terrain_grid(grid_size_m=5.0, resolution_cm=3.0)
            return height_grid
            
        except Exception as e:
            print(f"Error getting terrain grid: {e}")
            return None
    
    def _analyze_height_grid(self, height_grid):
        """Analyze the height grid to extract useful features
        
        Args:
            height_grid: 2D numpy array of height values
            
        Returns:
            Dictionary with analysis results
        """
        # Grid dimensions
        height, width = height_grid.shape
        
        # Calculate basic statistics
        mean_height = float(np.mean(height_grid))
        min_height = float(np.min(height_grid))
        max_height = float(np.max(height_grid))
        std_dev = float(np.std(height_grid))
        
        # Calculate gradient (slope) in x and y directions
        gradient_y, gradient_x = np.gradient(height_grid)
        
        # Calculate slope magnitude
        slope_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        max_slope = float(np.max(slope_magnitude))
        mean_slope = float(np.mean(slope_magnitude))
        
        # Detect obstacles (areas with significant height changes)
        # Here we consider a point an obstacle if the slope is steeper than 30 degrees
        # tan(30°) ≈ 0.577 for a 3cm grid
        obstacle_threshold = 0.577 * (3.0 / 100.0)  # Convert to slope per grid cell
        obstacles = slope_magnitude > obstacle_threshold
        obstacle_count = int(np.sum(obstacles))
        
        # Identify flat regions (low slope areas)
        flat_threshold = 0.1 * (3.0 / 100.0)  # 10% grade or less
        flat_regions = slope_magnitude < flat_threshold
        flat_area_percentage = float(np.sum(flat_regions) / (height * width) * 100)
        
        # Compute roughness (local variation)
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(height_grid, size=3)
        roughness = np.mean(np.abs(height_grid - local_mean))
        
        # Return analysis results
        return {
            "mean_height_m": mean_height,
            "min_height_m": min_height,
            "max_height_m": max_height,
            "height_range_m": max_height - min_height,
            "height_std_dev_m": std_dev,
            "max_slope": max_slope,
            "mean_slope": mean_slope,
            "obstacle_count": obstacle_count,
            "flat_area_percentage": flat_area_percentage,
            "roughness": float(roughness),
            "traversability_assessment": self._assess_traversability(max_slope, obstacle_count, roughness)
        }
    
    def _assess_traversability(self, max_slope, obstacle_count, roughness):
        """Assess overall traversability of the terrain
        
        Args:
            max_slope: Maximum slope value
            obstacle_count: Number of detected obstacles
            roughness: Terrain roughness value
            
        Returns:
            Dictionary with traversability assessment
        """
        # Convert slope to degrees for easier interpretation
        max_slope_degrees = np.arctan(max_slope) * 180 / np.pi
        
        # Define thresholds
        slope_threshold_easy = 15.0  # degrees
        slope_threshold_medium = 25.0  # degrees
        slope_threshold_hard = 35.0  # degrees
        
        obstacle_threshold_easy = 10
        obstacle_threshold_medium = 50
        obstacle_threshold_hard = 200
        
        roughness_threshold_easy = 0.01  # meters
        roughness_threshold_medium = 0.03  # meters
        roughness_threshold_hard = 0.05  # meters
        
        # Determine difficulty based on criteria
        if (max_slope_degrees > slope_threshold_hard or 
            obstacle_count > obstacle_threshold_hard or 
            roughness > roughness_threshold_hard):
            difficulty = "difficult"
            recommendation = "not recommended for traversal"
        elif (max_slope_degrees > slope_threshold_medium or 
              obstacle_count > obstacle_threshold_medium or 
              roughness > roughness_threshold_medium):
            difficulty = "moderate"
            recommendation = "proceed with caution"
        elif (max_slope_degrees > slope_threshold_easy or 
              obstacle_count > obstacle_threshold_easy or 
              roughness > roughness_threshold_easy):
            difficulty = "easy"
            recommendation = "safe to traverse"
        else:
            difficulty = "very easy"
            recommendation = "optimal for traversal"
        
        return {
            "difficulty": difficulty,
            "recommendation": recommendation,
            "reasons": {
                "slope": f"Max slope is {max_slope_degrees:.1f}° ({difficulty})",
                "obstacles": f"{obstacle_count} obstacles detected ({difficulty})",
                "roughness": f"Terrain roughness is {roughness:.3f}m ({difficulty})"
            }
        }
