import requests
import numpy as np
import time
import os
import json
import bosdyn.api.image_pb2 as image_pb2
from PIL import Image
from mimetypes import guess_type
from bosdyn.client import create_standard_sdk
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.frame_helpers import get_vision_tform_body, VISION_FRAME_NAME, BODY_FRAME_NAME
from bosdyn.client import frame_helpers
from bosdyn.client import ray_cast
from bosdyn.api import geometry_pb2
import io
from utils.timestamp_utils import parse_timestamp
from perception_utils import ObjectDetector
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.api import ray_cast_pb2

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
        
        # Cache for perception data to optimize background logging
        self.last_odometry_time = 0
        self.last_vision_time = 0
        self.cached_odometry = None
        self.cached_vision = None
        self.cache_timeout = 0.5  # seconds
        
        # Initialize vision model
        self.vision_model = None
        self.vision_processor = None
        
        # Initialize object detector
        self.object_detector = None
        
        self._image_client = None
        self._ray_cast_client = None
        self.yolo_api_url = os.getenv("YOLO_API_URL", "http://134.245.232.230:8002")  # Default local address
        
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
            
            # Initialize the object detector with the image client
            self.object_detector = ObjectDetector(self.image_client)
            
            # Power on the robot
            self.robot.power_on(timeout_sec=20)
            assert self.robot.is_powered_on(), "Robot power on failed"
            
            self._image_client = self.robot.ensure_client(ImageClient.default_service_name)
            self._ray_cast_client = self.robot.ensure_client(RayCastClient.default_service_name)
            print("Image and RayCast clients created.")
            
            self.connected = True
            return True
        except Exception as e:
            print(f"Error during connection or client creation: {e}")
            return False
    
    def get_odometry(self):
        """Retrieve the robot's current position and orientation"""
        if not self.connected:
            # Return simulated data in simulation mode
            odometry_data = {
                "position": self.position,
                "orientation": self.orientation
            }
        else:
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
                
                odometry_data = {
                    "position": self.position,
                    "orientation": self.orientation
                }
            except Exception as e:
                print(f"Error getting odometry: {e}")
                # Return the last known position in case of error
                odometry_data = {
                    "position": self.position,
                    "orientation": self.orientation
                }
        
        return odometry_data
    

        
    def analyze_images(self):
        """Capture images and generate a caption of what the robot sees"""

        # First capture images
        camera_names = ["frontleft_fisheye_image", "frontright_fisheye_image", "left_fisheye_image", "right_fisheye_image", "back_fisheye_image"]
        image_paths = self.take_pictures(camera_names)
        vision_data = []
        camera_name_to_description = {
            "frontleft_fisheye_image": "Ahead, to the right the",
            "frontright_fisheye_image": "Ahead, to the left the",
            "left_fisheye_image": "To the left the",
            "right_fisheye_image": "To the right the",
            "back_fisheye_image": "Behind the",
        }

        if not image_paths:
            #print("Failed to capture image")
            vision_data = {"description": "Unable to capture image"}
        else:
            # Use online vision model to analyze the image
            try:
                url = "http://134.245.232.230:8000/caption"
                for i, image_path in enumerate(image_paths):
                    with open(image_path, "rb") as f:
                        response = requests.post(
                            url,    
                            files = {
                            "file": (f.name, f, guess_type(image_path)[0] or "image/jpeg")
                            }
                        )
  
                    if response.status_code == 200:
                        caption = response.json()["caption"]
                    else:
                        raise
                    # Create description
                    description = f"{camera_name_to_description[camera_names[i]]} robot sees: {caption}"
                    
                    vision_data.append({"description": description})
                
            except Exception as e:
                vision_data = {"description": f"Image captioning currently unavailable"}
    
        return vision_data
    
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

    def take_pictures(self, camera_names=["frontleft_fisheye_image", "frontright_fisheye_image"]):
        """Capture an image from the robot's front camera"""
        if not self.connected:
            # In simulation mode, use a placeholder timestamp
            self.last_image_path = f"images/spot_image_sim_{int(time.time())}.jpg"
            
            # Create a simple test image
            img = Image.new('RGB', (640, 480), color=(73, 109, 137))
            img = img.rotate(90, expand=True)  # Rotate simulation image too
            img.save(self.last_image_path)
            return self.last_image_path
        
        try:
            # Get image client
            image_client : ImageClient = self.robot.ensure_client('image')
            
            # Request image from front camera
            request = [build_image_request(source, image_format=image_pb2.Image.Format.FORMAT_JPEG, pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8) for source in camera_names]
            image_response = image_client.get_image(request)
            
            # Save the image
            image_paths = []
            for i, image_data in enumerate(image_response): 
                self.last_image_path = f"images/spot_image_{camera_names[i]}_{int(time.time())}.jpg"
                
                # Convert image data to PIL Image
                image = Image.open(io.BytesIO(image_data.shot.image.data))
                
                # Apply rotations based on camera name
                if "frontleft" in camera_names[i] or "frontright" in camera_names[i]:
                    image = image.rotate(-90, expand=True)  # -90 for clockwise rotation
                elif "right" in camera_names[i]:
                    image = image.rotate(180, expand=True)
                
                # Save the rotated image
                image.save(self.last_image_path, quality=95)
                image_paths.append(self.last_image_path)
            return image_paths
        except Exception as e:
            print(f"Error taking pictures: {e}")
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

    def get_perception_logs(self, log_type=None, seconds=1):
        """Retrieve recent perception logs
        
        Args:
            log_type: Type of logs to retrieve ('odometry', 'vision', or None for both)
            seconds: Number of seconds to look back for perception logs
            
        Returns:
            Dictionary with most recent log entries
        """
        logs_dir = "perception_logs"
        
        if not os.path.exists(logs_dir):
            return {"error": "No perception logs found"}
        
        result = {}
        current_time = time.time()
        
        try:
            # Get odometry logs if requested
            if log_type is None or log_type == 'odometry':
                odometry_logs = sorted(
                    [f for f in os.listdir(logs_dir) if f.startswith('odometry_')],
                    reverse=True
                )
                
                if odometry_logs:
                    latest_log = odometry_logs[0]
                    log_path = os.path.join(logs_dir, latest_log)
                    
                    # Read the last 'count' lines
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        entries = []
                        for line in lines[-1:]:
                            entry = json.loads(line)
                            # Add time_ago field to each entry
                            if 'timestamp' in entry:
                                timestamp = parse_timestamp(entry['timestamp'], current_time)
                                entry['time_ago'] = round(current_time - timestamp, 1)
                            entries.append(entry)
                        result['odometry'] = entries
            
            # Get vision logs if requested
            if log_type is None or log_type == 'vision':
                vision_logs = sorted(
                    [f for f in os.listdir(logs_dir) if f.startswith('vision_')],
                    reverse=True
                )
                
                if vision_logs:
                    latest_log = vision_logs[0]
                    log_path = os.path.join(logs_dir, latest_log)
                    
                    # Read the last 'count' lines
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        entries = []
                        for line in lines[-1:]:
                            entry = json.loads(line)
                            # Add time_ago field to each entry
                            if 'timestamp' in entry:
                                timestamp = parse_timestamp(entry['timestamp'], current_time)
                                entry['time_ago'] = round(current_time - timestamp, 1)
                            entries.append(entry)
                        result['vision'] = entries
                        
            return result
            
        except Exception as e:
            print(f"Error retrieving perception logs: {e}")
            return {"error": str(e)}

    def get_action_logs(self, task_id=None, count=10):
        """Retrieve action logs
        
        Args:
            task_id: Specific task ID to retrieve (None for most recent)
            count: Number of log entries to retrieve
            
        Returns:
            Dictionary with task logs
        """
        logs_dir = "action_logs"
        
        if not os.path.exists(logs_dir):
            return {"error": "No action logs found"}
        
        try:
            # Get list of all task log files
            task_logs = sorted(
                [f for f in os.listdir(logs_dir) if f.startswith('task_')],
                reverse=True
            )
            
            if not task_logs:
                return {"error": "No task logs found"}
                
            # If task_id is specified, find that specific log
            if task_id:
                matching_logs = [log for log in task_logs if f"task_{task_id}_" in log]
                if matching_logs:
                    log_file = matching_logs[0]
                else:
                    return {"error": f"No task log found with ID {task_id}"}
            else:
                # Otherwise use the most recent log
                log_file = task_logs[0]
                # Extract task_id from filename
                task_id = log_file.split("_")[1]
            
            log_path = os.path.join(logs_dir, log_file)
            
            # Read the log file
            with open(log_path, 'r') as f:
                lines = f.readlines()
                
            # Parse all entries
            entries = [json.loads(line) for line in lines]
            
            # Return full task info
            return {
                "task_id": task_id,
                "filename": log_file,
                "total_entries": len(entries),
                "entries": entries[-count:] if count < len(entries) else entries,
                "has_more": len(entries) > count
            }
                
        except Exception as e:
            print(f"Error retrieving action logs: {e}")
            return {"error": str(e)}

    def get_odometry_logs(self, seconds=1):
        """Retrieve recent odometry logs
        
        Args:
            seconds: Number of seconds to look back for odometry logs
            
        Returns:
            Dictionary with most recent odometry log entries
        """
        logs = self.get_perception_logs('odometry', seconds)
        
        # Return just the odometry part
        if 'odometry' in logs:
            return {'odometry': logs['odometry']}
        else:
            return logs  # Returns error if any

    def get_vision_logs(self, seconds=1):
        """Retrieve recent vision logs
        
        Args:
            seconds: Number of seconds to look back for vision logs
            
        Returns:
            Dictionary with most recent vision log entries
        """
        logs = self.get_perception_logs('vision', seconds)
        
        # Return just the vision part
        if 'vision' in logs:
            return {'vision': logs['vision']}
        else:
            return logs  # Returns error if any

    def get_terrain_grid(self, grid_size_m=5.0, resolution_cm=3.0):
        """Get terrain height grid data around the robot
        
        Args:
            grid_size_m: Size of the grid in meters (default: 5.0)
            resolution_cm: Resolution of the grid in centimeters (default: 3.0)
            
        Returns:
            numpy.ndarray: 2D grid of height values, centered on the robot
        """
        if not self.connected:
            # Simulation mode - create a synthetic terrain grid
            return self._simulate_terrain_grid(grid_size_m, resolution_cm)
        
        try:
            # In a real implementation, this would use the Spot API to get terrain data
            # This could use several approaches:
            # 1. Use the depth images from multiple cameras to create a local height map
            # 2. Request terrain data from the robot's SLAM or mapping system
            # 3. Use a separate lidar or depth sensor data

            # For now, since we don't have direct access to height grid data in the Spot API,
            # we'll use the simulation as a placeholder until a real implementation is created
            return self._simulate_terrain_grid(grid_size_m, resolution_cm)
        
        except Exception as e:
            print(f"Error getting terrain grid: {e}")
            # Fall back to simulation on error
            return self._simulate_terrain_grid(grid_size_m, resolution_cm)
        
    def _simulate_terrain_grid(self, grid_size_m=5.0, resolution_cm=3.0):
        """Create a simulated terrain grid for testing
        
        Args:
            grid_size_m: Size of the grid in meters
            resolution_cm: Resolution of the grid in centimeters
            
        Returns:
            numpy.ndarray: 2D grid of simulated height values
        """
        import numpy as np
        
        # Calculate grid dimensions based on size and resolution
        grid_cells = int(grid_size_m * 100 / resolution_cm)
        
        # Get current robot position and orientation
        position = self.position
        orientation = self.orientation
        
        # Create base grid - flat terrain with small random noise
        height_grid = np.zeros((grid_cells, grid_cells))
        height_grid += np.random.normal(0, 0.01, height_grid.shape)  # 1cm noise
        
        # Create coordinate grid
        x, y = np.meshgrid(
            np.linspace(-grid_size_m/2, grid_size_m/2, grid_cells),
            np.linspace(-grid_size_m/2, grid_size_m/2, grid_cells)
        )
        
        # Add some terrain features based on robot position to make it change as robot moves
        
        # Add a small hill
        hill_x = position["x"] % 2 - 1  # Use modulo to create repeating pattern
        hill_y = position["y"] % 2 - 1
        r = np.sqrt((x - hill_x)**2 + (y - hill_y)**2)
        height_grid += 0.15 * np.exp(-r**2 / 1.0)  # 15cm tall Gaussian hill
        
        # Add a ditch/trench
        trench_angle = np.radians(orientation["yaw"] + 45)  # Angled relative to robot orientation
        trench_dist = x * np.cos(trench_angle) + y * np.sin(trench_angle)
        height_grid -= 0.1 * np.exp(-trench_dist**2 / 0.04)  # 10cm deep trench
        
        # Add some small random obstacles
        for _ in range(10):
            # Random positions within the grid
            obs_x = np.random.uniform(-grid_size_m/2, grid_size_m/2)
            obs_y = np.random.uniform(-grid_size_m/2, grid_size_m/2)
            obs_r = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            # Small obstacles of varying height (2-5cm)
            obs_height = np.random.uniform(0.02, 0.05)
            height_grid += obs_height * np.exp(-obs_r**2 / 0.01)
        
        # Add a simulated wall or edge beyond which height increases rapidly
        edge_dist = 2.0  # 2 meters from center
        wall_mask = (np.abs(x) > edge_dist) | (np.abs(y) > edge_dist)
        height_grid[wall_mask] += 0.2  # 20cm wall/edge
        
        # Ensure all heights are physically plausible
        # (e.g., if we're simulating terrain relative to robot standing height)
        height_grid += 0.01  # Add 1cm minimum ground clearance
        
        return height_grid

    def get_terrain_logs(self, seconds=1):
        """Retrieve recent terrain analysis logs
        
        Args:
            seconds: Number of seconds to look back for terrain logs
            
        Returns:
            Dictionary with most recent terrain log entries
        """
        logs_dir = "perception_logs"
        
        if not os.path.exists(logs_dir):
            return {"error": "No terrain logs found"}
        
        result = {}
        
        try:
            # Get terrain logs
            terrain_logs = sorted(
                [f for f in os.listdir(logs_dir) if f.startswith('terrain_')],
                reverse=True
            )
            
            if terrain_logs:
                latest_log = terrain_logs[0]
                log_path = os.path.join(logs_dir, latest_log)
                
                # Read the last N lines
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    # Terrain logs are less frequent, so we need fewer entries per second
                    entries = [json.loads(line) for line in lines[-seconds*2:]]
                    result['terrain'] = entries
                    
            return result
            
        except Exception as e:
            print(f"Error retrieving terrain logs: {e}")
            return {"error": str(e)}

    def locate_objects_in_view(self, target_frame=frame_helpers.VISION_FRAME_NAME):
        """
        Captures images from all cameras, detects objects using YOLO, and locates them in 3D space.
        
        Args:
            target_frame (str): The frame to report object locations in.
            
        Returns:
            tuple: (located_objects, error_message)
                   located_objects: List of objects with 3D positions, or None on error.
                   error_message: String description of the error, or None on success.
        """
        
        # All available camera sources
        camera_sources = [
            "frontleft_fisheye_image", 
            "frontright_fisheye_image", 
            "left_fisheye_image", 
            "right_fisheye_image", 
            "back_fisheye_image"
        ]
        
        all_located_objects = []
        
        for camera_source in camera_sources:
            
            # 1. Get Image and Metadata
            image_response, error = self.get_image_and_metadata(camera_source)
            if error:
                print(f"Error getting image from {camera_source}: {error}. Skipping this camera.")
                continue
                
            # 2. Detect Objects via YOLO
            detections, error = self.detect_objects_in_image(image_response)
            if error:
                print(f"Error detecting objects in {camera_source}: {error}. Skipping this camera.")
                continue
                
            if not detections:
                continue
                
            # 3. Project Detections to 3D
            located_objects, error = self.get_object_locations(detections, image_response, target_frame)
            if error:
                print(f"Error locating objects in 3D from {camera_source}: {error}. Skipping this camera.")
                continue
                
            # Add source camera information to each object
            for obj in located_objects:
                obj['source_camera'] = camera_source
                
            # Add to aggregated results
            all_located_objects.extend(located_objects)
            
        # Summarize results
        if all_located_objects:
            # Group by label
            label_counts = {}
            for obj in all_located_objects:
                label = obj['label']
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
                    
            summary = ", ".join([f"{count} {label}{'s' if count > 1 else ''}" for label, count in label_counts.items()])
            return all_located_objects, None
        else:
            return [], None  # Return empty list, not an error

    def get_image_and_metadata(self, source_name="frontleft_fisheye_image"):
        """
        Captures an image from a specified source along with its metadata.

        Args:
            source_name (str): The name of the image source (e.g., 'frontleft_fisheye_image').

        Returns:
            tuple: (image_response, error_message)
                   image_response: The ImageResponse object from the SDK, or None on error.
                   error_message: String description of the error, or None on success.
        """
        if not self._image_client:
            return None, "Image client not initialized."
        try:
            # Request an uncompressed RGB image
            image_request = build_image_request(
                source_name, 
                quality_percent=100,
                image_format=image_pb2.Image.FORMAT_JPEG,
                pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8
            )

            image_responses = self._image_client.get_image([image_request])

            if len(image_responses) < 1:
                return None, "No image received from robot."

            image_response = image_responses[0]  # Assuming one source requested

            # Check for basic validity (has shot, transform snapshot, intrinsics)
            if not image_response.shot.transforms_snapshot:
                return None, f"Image source {source_name} missing transform snapshot."
            if not image_response.source.pinhole:
                return None, f"Image source {source_name} missing pinhole camera intrinsics."

            return image_response, None

        except Exception as e:
            return None, f"Error getting image from {source_name}: {e}"
    
    def detect_objects_in_image(self, image_response):
        """
        Sends image data to the YOLO API and returns detections.

        Args:
            image_response: The ImageResponse object containing image data.

        Returns:
            tuple: (list_of_detections, error_message)
                   list_of_detections: A list of dicts [{'label': str, 'score': float, 'box': [x_min, y_min, x_max, y_max]}, ...], 
                   or None on error.
                   error_message: String description of the error, or None on success.
        """
        if not image_response or not image_response.shot.image.data:
            return None, "Invalid image_response provided."

        try:
            
            # For RAW format, we'd need to convert to a recognizable image format
            # In a production system, we'd decode the RAW format based on its properties
            # For simplicity in this example, we'll assume the YOLO API can handle common formats
            image_bytes = io.BytesIO(image_response.shot.image.data)
            
            files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}  # Adjust filename/mimetype as needed
            params = {'threshold': 0.5}  # Adjust threshold as needed

            start_time = time.time()
            response = requests.post(f"{self.yolo_api_url}/detect", files=files, params=params, timeout=10)
            end_time = time.time()

            response.raise_for_status()  # Raise an exception for bad status codes

            results = response.json()
            detections = results.get("detections", [])

            # Basic validation of detection format
            validated_detections = []
            for det in detections:
                if isinstance(det, dict) and 'label' in det and 'score' in det and 'box' in det and len(det['box']) == 4:
                    validated_detections.append(det)
                else:
                    print(f"Warning: Skipping invalid detection format: {det}")

            return validated_detections, None

        except requests.exceptions.RequestException as e:
            return None, f"Error calling YOLO API: {e}"
        except ValueError as e:  # Includes JSONDecodeError
            return None, f"Error decoding JSON response from YOLO API: {e}"
        except Exception as e:
            return None, f"Unexpected error during YOLO detection: {e}"
    
    def get_object_locations(self, detections, image_response, target_frame_name=frame_helpers.BODY_FRAME_NAME):
        """
        Projects 2D detection bounding box centers to 3D points using ray casting.

        Args:
            detections (list): List of detections [{'label':..., 'score':..., 'box':...}, ...].
            image_response: The ImageResponse containing metadata (transforms, camera).
            target_frame_name (str): The desired frame for the 3D points.

        Returns:
            tuple: (located_objects, error_message)
                   located_objects: List of dicts with 3D positions added, or None on error.
                   error_message: String description of the error, or None on success.
        """
        if not self._ray_cast_client:
            return None, "RayCast client not initialized."
        if not detections or not image_response:
            return None, "Missing detections or image_response."

        located_objects = []
        transforms_snapshot = image_response.shot.transforms_snapshot
        camera_intrinsics = image_response.source.pinhole.intrinsics
        frame_name_image_sensor = image_response.shot.frame_name_image_sensor

        try:
            # Get transform from camera to target frame
            camera_tform_target = frame_helpers.get_a_tform_b(
                transforms_snapshot,
                frame_name_image_sensor,
                target_frame_name
            )
            
            if not camera_tform_target:
                return None, f"Could not get transform from {frame_name_image_sensor} to {target_frame_name}"
            
            # Get target tform camera (inverse transform)
            target_tform_camera = camera_tform_target.inverse()
            
            # Camera position in target frame
            camera_position = target_tform_camera.get_translation()
            
            # Ray cast intersection types (empty = all types)
            raycast_types = []
            
            # Cache camera name for debugging
            camera_name = frame_name_image_sensor
            
            for detection in detections:
                box = detection['box']  # [x_min, y_min, x_max, y_max]
                center_px_x = (box[0] + box[2]) / 2
                center_px_y = (box[1] + box[3]) / 2
                
                # Calculate normalized coordinates using camera intrinsics
                focal_x = camera_intrinsics.focal_length.x
                focal_y = camera_intrinsics.focal_length.y
                principal_x = camera_intrinsics.principal_point.x
                principal_y = camera_intrinsics.principal_point.y
                
                norm_x = (center_px_x - principal_x) / focal_x
                norm_y = (center_px_y - principal_y) / focal_y
                
                # Create ray direction in camera frame
                # For pinhole camera, Z is forward, X is right, Y is down
                ray_dir_camera = np.array([norm_x, norm_y, 1.0])
                
                # Normalize the ray direction
                ray_dir_camera = ray_dir_camera / np.linalg.norm(ray_dir_camera)
                
                # Important: Use ONLY the rotation component to transform the direction vector
                # This is critical for correctly handling camera orientations
                # The SE3Pose rotation component can be accessed via the .rotation property
                ray_dir_target = target_tform_camera.rotation.transform_point(
                    ray_dir_camera[0], ray_dir_camera[1], ray_dir_camera[2]
                )
                
                # Make sure the direction is normalized
                magnitude = np.sqrt(ray_dir_target[0]**2 + ray_dir_target[1]**2 + ray_dir_target[2]**2)
                ray_dir_target = [
                    ray_dir_target[0]/magnitude,
                    ray_dir_target[1]/magnitude, 
                    ray_dir_target[2]/magnitude
                ]
                
                try:
                    # Cast the ray from camera position along the transformed direction
                    ray_results = self._ray_cast_client.raycast(
                        ray_origin=camera_position,
                        ray_direction=ray_dir_target,
                        raycast_types=raycast_types,
                        frame_name=target_frame_name
                    )
                    
                    if not ray_results.hits:
                        continue
                        
                    # Get the closest hit
                    hit = ray_results.hits[0]
                    hit_position = hit.hit_position_in_hit_frame
                    
                    # Add to located objects
                    located_objects.append({
                        'label': detection['label'],
                        'score': detection['score'],
                        'box': detection['box'],
                        'position': {
                            'x': hit_position.x,
                            'y': hit_position.y,
                            'z': hit_position.z
                        },
                        'hit_type': hit.type,
                        'source_camera': camera_name
                    })
                except Exception as e:
                    print(f"Error ray casting for object '{detection['label']}' from {camera_name}: {e}")
                    continue

            return located_objects, None

        except Exception as e:
            return None, f"Error during 2D-to-3D projection: {e}"

    def get_object_detection_logs(self, seconds=1):
        """Retrieve recent object detection logs
        
        Args:
            seconds: Number of seconds to look back for object detection logs
            
        Returns:
            Dictionary with most recent object detection entries
        """
        logs_dir = "perception_logs"
        
        if not os.path.exists(logs_dir):
            return {"error": "No object detection logs found"}
        
        result = {}
        current_time = time.time()
        
        try:
            # Get object detection logs
            object_logs = sorted(
                [f for f in os.listdir(logs_dir) if f.startswith('objects_')],
                reverse=True
            )
            
            if object_logs:
                latest_log = object_logs[0]
                log_path = os.path.join(logs_dir, latest_log)
                
                # Read the last 'count' lines based on seconds parameter
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    entries = []
                    # Since object detection is less frequent, we need fewer entries
                    # Assuming 1 entry per 1-2 seconds
                    for line in lines[-seconds:]:
                        entry = json.loads(line)
                        # Add time_ago field to each entry
                        if 'timestamp' in entry:
                            timestamp = parse_timestamp(entry['timestamp'], current_time)
                            entry['time_ago'] = round(current_time - timestamp, 1)
                        entries.append(entry)
                    result['objects'] = entries
                    
            return result
            
        except Exception as e:
            print(f"Error retrieving object detection logs: {e}")
            return {"error": str(e)}
