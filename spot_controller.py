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
from bosdyn.client.frame_helpers import get_vision_tform_body
import io
from utils.timestamp_utils import parse_timestamp

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
                            "image_file": (f.name, f, guess_type(image_path)[0] or "image/jpeg")
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
                print(f"Error analyzing image with online vision model: {e}")
                vision_data = {"description": f"Image analysis failed due to an error: {str(e)}"}
    
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
