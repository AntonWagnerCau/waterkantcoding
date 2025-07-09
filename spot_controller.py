import requests
import numpy as np
import time
import os
import json
import bosdyn.api.image_pb2 as image_pb2
from PIL import Image, ImageDraw
from mimetypes import guess_type
from bosdyn.client import create_standard_sdk
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client.image import ImageClient, build_image_request, pixel_to_camera_space
from bosdyn.client.frame_helpers import get_vision_tform_body, VISION_FRAME_NAME, BODY_FRAME_NAME
from bosdyn.client import frame_helpers
from bosdyn.client import ray_cast
from bosdyn.api import geometry_pb2
import io
from utils.timestamp_utils import parse_timestamp
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.api import ray_cast_pb2
import math # Import math for degrees conversion
from bosdyn.api import geometry_pb2
from bosdyn.geometry import EulerZXY
import time

# Helper function for coordinate transformation
def transform_point_for_rotation(px, py, orig_w, orig_h, rot_w, rot_h, angle_deg):
    """Transforms a point from rotated image coords back to original image coords."""
    # Center coordinates relative to image center
    cx_rot = px - rot_w / 2.0
    cy_rot = py - rot_h / 2.0

    # Inverse rotation angle in radians
    angle_rad = math.radians(-angle_deg) # Negative angle for inverse
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Apply inverse rotation matrix
    cx_orig_rel = cx_rot * cos_a - cy_rot * sin_a
    cy_orig_rel = cx_rot * sin_a + cy_rot * cos_a

    # Convert back to top-left origin coordinates in the original frame
    cx_orig = cx_orig_rel + orig_w / 2.0
    cy_orig = cy_orig_rel + orig_h / 2.0

    return cx_orig, cy_orig

class SpotController:
    """Controls the Boston Dynamics Spot robot"""
    def __init__(self):
        self.connected = False
        self.robot = None
        self.command_client = None
        self.state_client = None
        self.image_client = None
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
        
        self.image_client = None
        self.ray_cast_client = None
        self.yolo_api_url = os.getenv("YOLO_API_URL", "http://134.245.232.230:8002")
        
    def connect(self):
        """Connect to the Spot robot"""
        try:
            sdk = create_standard_sdk("SpotAgentClient")
            spot_ip = os.getenv("SPOT_IP")
            if not spot_ip:
                print("Error: SPOT_IP environment variable not set.")
                return False
            self.robot = sdk.create_robot(spot_ip)
            
            username = os.getenv("SPOT_USERNAME")
            password = os.getenv("SPOT_PASSWORD")
            if not username or not password:
                 print("Error: SPOT_USERNAME or SPOT_PASSWORD environment variables not set.")
                 return False
            self.robot.authenticate(username, password)
            
            self.robot.time_sync.wait_for_sync()
            
            self.lease_client : LeaseClient = self.robot.ensure_client('lease')
            self.lease = self.lease_client.take()
            
            self.command_client : RobotCommandClient = self.robot.ensure_client(RobotCommandClient.default_service_name)
            self.state_client : RobotStateClient = self.robot.ensure_client(RobotStateClient.default_service_name)
            self.image_client : ImageClient = self.robot.ensure_client(ImageClient.default_service_name)
            self.ray_cast_client : RayCastClient = self.robot.ensure_client(RayCastClient.default_service_name)
            print("Base clients created.")

            self.robot.power_on(timeout_sec=20)
            assert self.robot.is_powered_on(), "Robot power on failed"
            print("Robot powered on.")

            self.connected = True
            print(f"Successfully connected to Spot at {spot_ip}")
            return True
        except Exception as e:
            print(f"Error connecting to Spot: {e}")
            self.connected = False
            return False
    
    def get_odometry(self):
        """Retrieve the robot's current position and orientation (in degrees)."""
        # In simulation mode, just return the last known/simulated state
        if not self.connected:
            return {
                "position": self.position,
                "orientation": self.orientation # Assume simulated orientation is already in degrees if needed
            }
            
        try:
            robot_state = self.state_client.get_robot_state()
            kinematic_state = robot_state.kinematic_state
            transforms = kinematic_state.transforms_snapshot
            
            # Get the transform from vision frame to body frame
            vision_tform_body = frame_helpers.get_vision_tform_body(transforms)
            
            # Extract position
            pos = vision_tform_body.position
            current_position = {"x": pos.x, "y": pos.y, "z": pos.z}
            
            # Extract rotation (quaternion) and convert roll, pitch, yaw to degrees
            rot = vision_tform_body.rotation
            current_orientation = {
                "roll": math.degrees(rot.to_roll()),
                "pitch": math.degrees(rot.to_pitch()),
                "yaw": math.degrees(rot.to_yaw())
            }
            
            # Update internal state (optional, depending on if simulation uses it)
            self.position = current_position
            self.orientation = current_orientation 
            
            return {
                "position": current_position,
                "orientation": current_orientation # Now in degrees
            }
            
        except Exception as e:
            print(f"Error getting odometry: {e}")
            # Return the last known state in case of error
            return {
                "position": self.position,
                "orientation": self.orientation 
            }
    
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

    def tilt(self, pitch=0.0, roll=0.0, yaw=0.0, bh=0.0):
        """Command the robot to tilt its body with specified roll, pitch, yaw angles (in radians), and adjust body height (in meters)."""
        if not self.connected:
            print(f"Robot not connected, simulating tilt: roll={roll}, pitch={pitch}, yaw={yaw}, bh={bh}")
            return True

        try:
            # Roll, pitch, yaw
            orientation = EulerZXY(roll=roll, pitch=pitch, yaw=yaw)

            # Create the stand command with tilted orientation and body height
            cmd = RobotCommandBuilder.synchro_stand_command(
                body_height=bh,
                footprint_R_body=orientation
            )

            # Send the command
            self.command_client.robot_command(cmd, end_time_secs=time.time() + 3)

            return True

        except Exception as e:
            print(f"Error tilting: {e}")
            return False

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

    def locate_objects_in_view(self, target_frame=frame_helpers.BODY_FRAME_NAME, return_images=False):
        """
        Captures images, rotates for upright view, detects objects on rotated image,
        projects detections back to original frame for 3D localization, 
        returns object locations, optionally annotated images, and camera details.

        Args:
            target_frame (str): The frame to report 3D object locations and camera poses in (e.g., body).
            return_images (bool): Whether to return annotated images (rotated) with bounding boxes.

        Returns:
            dict: {
                'objects': list_of_objects_with_3D_pos,
                'images': dict {camera_name: {'annotated_image': rotated_pil_image_with_boxes }} (if return_images=True),
                'camera_details': dict {camera_name: {'intrinsics': {...}, 'pose': {...}, 'dimensions': {...}}},
                'error': error message (if any)
            }
        """
        camera_sources = [
            "frontleft_fisheye_image", "frontright_fisheye_image", 
            "left_fisheye_image", "right_fisheye_image", "back_fisheye_image"
        ]
        
        all_located_objects = [] # Store final objects with 3D positions
        images_for_return = {} if return_images else None
        camera_details = {} # Store camera pose and intrinsics

        try:
            for camera_source in camera_sources:
                # 1. Get Original Image and Metadata
                image_response, error = self.get_image_and_metadata(camera_source)
                if error or not image_response: # Check image_response explicitly
                    print(f"Skipping camera {camera_source} due to error: {error}")
                    continue
                
                # Extract camera details here, even if detection fails later
                try:
                    intrinsics = image_response.source.pinhole.intrinsics
                    transforms = image_response.shot.transforms_snapshot
                    frame_name_image_sensor = image_response.shot.frame_name_image_sensor
                    
                    # Get transform representing the sensor frame's pose IN the target (body) frame
                    tform_sensor_in_body = frame_helpers.get_a_tform_b(transforms, target_frame, frame_name_image_sensor)
                    
                    if tform_sensor_in_body:
                        pos = tform_sensor_in_body.position
                        rot = tform_sensor_in_body.rotation # Quaternion
                        
                        camera_details[camera_source] = {
                            'intrinsics': {
                                'focal_length_x': intrinsics.focal_length.x,
                                'focal_length_y': intrinsics.focal_length.y,
                                'principal_point_x': intrinsics.principal_point.x,
                                'principal_point_y': intrinsics.principal_point.y,
                            },
                            'pose': { # Pose of camera in target_frame (body)
                                'position': {'x': pos.x, 'y': pos.y, 'z': pos.z},
                                'rotation': {'w': rot.w, 'x': rot.x, 'y': rot.y, 'z': rot.z}
                            },
                            'dimensions': {
                                'width': image_response.shot.image.cols,
                                'height': image_response.shot.image.rows
                            }
                        }
                    else:
                        print(f"Warning: Could not get transform for {camera_source} to {target_frame}")

                except Exception as cam_detail_err:
                    print(f"Error extracting camera details for {camera_source}: {cam_detail_err}")
                    # Continue processing the image for detection if possible

                original_image_data = image_response.shot.image # Keep original proto
                orig_w = original_image_data.cols
                orig_h = original_image_data.rows

                # 2. Create PIL Image & Apply Rotation for Detection View
                try:
                    pil_image_for_detection = Image.open(io.BytesIO(original_image_data.data))
                    rotation_angle = 0 # Default no rotation

                    # Determine rotation based on camera source
                    if camera_source == "frontleft_fisheye_image" or camera_source == "frontright_fisheye_image":
                        rotation_angle = -90
                    elif camera_source == "right_fisheye_image":
                        rotation_angle = 180
                    # Add other rotations if needed

                    # Apply rotation if necessary
                    if rotation_angle != 0:
                        pil_image_for_detection = pil_image_for_detection.rotate(rotation_angle, expand=True)

                except Exception as img_proc_err:
                     print(f"Error processing image for {camera_source}: {img_proc_err}")
                     continue # Skip detection and projection for this camera

                # 3. Detect Objects on the (potentially) Rotated Image
                detections, error = self.detect_objects_in_image(pil_image_for_detection)
                if error:
                    print(f"Error detecting objects in {camera_source}: {error}")
                    # If error and return_images, add the (potentially rotated) empty image
                    if return_images:
                        images_for_return[camera_source] = {'annotated_image': pil_image_for_detection}
                    continue
                
                # If no detections, add the (potentially rotated) empty image if needed and continue
                if not detections and return_images:
                    images_for_return[camera_source] = {'annotated_image': pil_image_for_detection}
                    continue # No detections to project

                # Process detections if they exist
                if detections:
                    # Prepare annotated image for return (optional)
                    if return_images:
                        try:
                            # Draw directly on the image used for detection (rotated)
                            annotated_image_rotated = pil_image_for_detection.copy()
                            draw = ImageDraw.Draw(annotated_image_rotated)
                            for detection in detections:
                                box = detection.get('box', [0, 0, 0, 0])
                                label = detection.get('label', 'unknown')
                                score = detection.get('score', 0.0)
                                # Coordinates are relative to the image passed to detector
                                draw.rectangle(box, outline='lime', width=3)
                                text = f"{label} ({score:.2f})"
                                text_pos = (box[0] + 3, box[1] + 3)
                                if box[1] > 15: text_pos = (box[0] + 3, box[1] - 15)
                                draw.text(text_pos, text, fill='lime')
                            
                            images_for_return[camera_source] = {'annotated_image': annotated_image_rotated}

                        except Exception as draw_err:
                            print(f"Error drawing boxes on {camera_source}: {draw_err}")
                            # Fallback to the unannotated (but potentially rotated) image used for detection
                            if camera_source not in images_for_return:
                                images_for_return[camera_source] = {'annotated_image': pil_image_for_detection}

                    # 4. Project to 3D using coordinates transformed back to ORIGINAL frame
                    rot_w, rot_h = pil_image_for_detection.size # Dimensions of the image used for detection
                    for det in detections:
                        box_rot = det['box'] # Box is relative to the image passed to detector
                        # Center point in the rotated image's coordinate system
                        center_x_rot = (box_rot[0] + box_rot[2]) / 2.0
                        center_y_rot = (box_rot[1] + box_rot[3]) / 2.0
                        
                        # Transform center point back to original image coordinates
                        if rotation_angle == -90:
                            # Use CORRECTED direct formula for -90deg CW rotation with expand=True
                            # px_orig = py_rot
                            # py_orig = orig_h - px_rot
                            center_x_orig = center_y_rot
                            center_y_orig = orig_h - center_x_rot
                        else:
                            # Use the general function for other rotations (or 0)
                            # WARNING: transform_point_for_rotation might be buggy for expand=True cases!
                            center_x_orig, center_y_orig = transform_point_for_rotation(
                                center_x_rot, center_y_rot,
                                orig_w, orig_h, rot_w, rot_h, rotation_angle
                            )

                        # 5. Perform Ray Casting using the transformed ORIGINAL coordinates and metadata
                        hit_point_3d = self._project_pixel_to_3d(center_x_orig, center_y_orig, image_response, target_frame)
                        
                        if hit_point_3d:
                            all_located_objects.append({
                                'label': det['label'],
                                'score': det['score'],
                                # 'box_rotated': box_rot, # Box relative to rotated image used for detection (optional)
                                'position': { # 3D position in target_frame
                                    'x': hit_point_3d.x,
                                    'y': hit_point_3d.y,
                                    'z': hit_point_3d.z
                                },
                                # 'hit_type': hit_point_3d.type, # If _project_pixel returns full hit
                                'source_camera': camera_source 
                            })
            
            # 6. Return structured dictionary
            return_data = {
                'objects': all_located_objects,
                'camera_details': camera_details,
            }
            if return_images:
                return_data['images'] = images_for_return
                
            return return_data
                
        except Exception as e:
            error_msg = f"Critical Error in locate_objects_in_view: {e}"
            print(error_msg)
            # Add traceback for debugging critical errors
            import traceback
            traceback.print_exc()
            # Return error structure
            return {'error': error_msg, 'objects': [], 'images': {}, 'camera_details': {}}

    def _project_pixel_to_3d(self, px, py, image_response, target_frame_name):
        """Helper function to project a single pixel coordinate to 3D using ray casting."""
        if not self.ray_cast_client:
            print("RayCast client not initialized.")
            return None
        if not image_response:
             print("Missing image_response for 3D projection.")
             return None
             
        transforms_snapshot = image_response.shot.transforms_snapshot
        image_source_proto = image_response.source # Use the ImageSource proto
        frame_name_image_sensor = image_response.shot.frame_name_image_sensor

        try:
            camera_tform_target = frame_helpers.get_a_tform_b(
                transforms_snapshot, frame_name_image_sensor, target_frame_name)
            if not camera_tform_target:
                 print(f"Could not get transform from {frame_name_image_sensor} to {target_frame_name}")
                 return None
            
            target_tform_camera = camera_tform_target.inverse()
            camera_position = target_tform_camera.get_translation()
            
            # Calculate ray direction in camera frame using pixel_to_camera_space
            # This returns a 3-tuple (x, y, z) in camera frame at 1m depth (direction vector)
            ray_dir_camera = pixel_to_camera_space(image_source_proto, px, py)
            
            # Transform ray direction to target frame using ONLY rotation
            ray_dir_target = target_tform_camera.rotation.transform_point(
                ray_dir_camera[0], ray_dir_camera[1], ray_dir_camera[2])
            
            # Normalize target direction vector
            magnitude = np.linalg.norm(ray_dir_target)
            if magnitude == 0:
                 print("Warning: Zero magnitude ray direction after transform.")
                 return None
            ray_dir_target_normalized = [d / magnitude for d in ray_dir_target]

            # Cast the ray
            ray_results = self.ray_cast_client.raycast(
                ray_origin=camera_position,
                ray_direction=ray_dir_target_normalized,
                frame_name=target_frame_name,
                raycast_types=[]
            )
                
            if ray_results.hits:
                # Return the 3D position of the closest hit
                return ray_results.hits[0].hit_position_in_hit_frame 
            else:
                # No hit found
                return None
                
        except Exception as e:
            print(f"Error during ray casting for pixel ({px},{py}) in {frame_name_image_sensor}: {e}")
            return None

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
        if not self.image_client:
            return None, "Image client not initialized."
        try:
            # Request an uncompressed RGB image
            image_request = build_image_request(
                source_name, 
                quality_percent=100,
                image_format=image_pb2.Image.FORMAT_JPEG,
                pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8
            )

            image_responses = self.image_client.get_image([image_request])
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
    
    def detect_objects_in_image(self, pil_image):
        """
        Sends image data to the YOLO API and returns detections.

        Args:
            pil_image: The PIL Image object containing image data.

        Returns:
            tuple: (list_of_detections, error_message)
                   list_of_detections: A list of dicts [{'label': str, 'score': float, 'box': [x_min, y_min, x_max, y_max]}, ...], 
                   or None on error.
                   error_message: String description of the error, or None on success.
        """

        try:
            
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format="JPEG", quality=100)
            img_bytes = img_bytes.getvalue()
            files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}  # Adjust filename/mimetype as needed
            params = {'threshold': 0.75}  # Adjust threshold as needed

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

            return validated_detections, None

        except requests.exceptions.RequestException as e:
            return None, f"Error calling YOLO API: {e}"
        except ValueError as e:  # Includes JSONDecodeError
            return None, f"Error decoding JSON response from YOLO API: {e}"
        except Exception as e:
            return None, f"Unexpected error during YOLO detection: {e}"
    
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
