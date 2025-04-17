import os
import threading
import time
import json
import numpy as np
from spot_controller import SpotController
from datetime import datetime

class PerceptionLogger:
    """Runs odometry and vision perception in background and logs data to files"""
    def __init__(self, spot_controller : SpotController):
        self.spot_controller = spot_controller
        self.running = False
        self.odometry_thread = None
        self.vision_thread = None
        self.terrain_thread = None
        self.object_detection_thread = None  # New thread for object detection
        self.odometry_interval = float(os.getenv("ODOMETRY_INTERVAL", "0.2"))  # seconds
        self.vision_interval = float(os.getenv("VISION_INTERVAL", "0.5"))  # seconds
        self.terrain_interval = float(os.getenv("TERRAIN_INTERVAL", "0.5"))  # seconds
        self.object_detection_interval = float(os.getenv("OBJECT_DETECTION_INTERVAL", "0.5"))  # seconds
        
        # Create logs directory if it doesn't exist
        self.logs_dir = "perception_logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Set up log file paths with ISO timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.odometry_log_path = os.path.join(self.logs_dir, f"odometry_{timestamp}.jsonl")
        self.vision_log_path = os.path.join(self.logs_dir, f"vision_{timestamp}.jsonl")
        self.terrain_log_path = os.path.join(self.logs_dir, f"terrain_{timestamp}.jsonl")
        self.object_detection_log_path = os.path.join(self.logs_dir, f"objects_{timestamp}.jsonl")  # New log file
        
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
        
        # Start terrain thread
        #self.terrain_thread = threading.Thread(target=self._terrain_loop)
        #self.terrain_thread.daemon = True
        #self.terrain_thread.start()
        
        # Start object detection thread
        self.object_detection_thread = threading.Thread(target=self._object_detection_loop)
        self.object_detection_thread.daemon = True
        self.object_detection_thread.start()
        
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
            
        if self.terrain_thread and self.terrain_thread.is_alive():
            self.terrain_thread.join(timeout=2.0)
        
        if self.object_detection_thread and self.object_detection_thread.is_alive():
            self.object_detection_thread.join(timeout=2.0)
            
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
            
    def _terrain_loop(self):
        """Background loop for terrain analysis"""
        print(f"Terrain logging started, interval: {self.terrain_interval}s")
        
        while self.running:
            try:
                # Get terrain data from controller
                terrain_data = self.analyze_terrain()
                
                if terrain_data:
                    # Log the data
                    self._log_data(self.terrain_log_path, terrain_data)
                
            except Exception as e:
                print(f"Error in terrain loop: {e}")
                
            # Sleep for the configured interval
            time.sleep(self.terrain_interval)
    
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

    def _object_detection_loop(self):
        """Background loop for object detection"""
        print(f"Object detection logging started, interval: {self.object_detection_interval}s")
        
        while self.running:
            try:
                # Run locate_objects_in_view method to detect and localize objects
                objects, error = self.spot_controller.locate_objects_in_view()
                
                # Create the detection results object
                if error:
                    detection_data = {
                        "status": "error",
                        "error": error
                    }
                else:
                    # Filter out low confidence detections
                    filtered_objects = []
                    for obj in objects:
                        # Include objects with score > 0.4
                        if obj.get('score', 0) > 0.4:
                            filtered_objects.append(obj)
                    
                    detection_data = {
                        "status": "success",
                        "objects": filtered_objects,
                        "object_count": len(filtered_objects)
                    }
                
                # Log the object detection data
                self._log_data(self.object_detection_log_path, detection_data)
                
            except Exception as e:
                print(f"Error in object detection loop: {e}")
                # Log the error
                self._log_data(self.object_detection_log_path, {
                    "status": "error",
                    "error": str(e)
                })
                
            # Sleep for the configured interval
            time.sleep(self.object_detection_interval)
