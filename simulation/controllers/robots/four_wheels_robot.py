from controller import Supervisor, Camera, LED, Motor
import sys
import os
import numpy as np
from math import radians, degrees, atan2, sqrt, pi
import time
from math import sqrt, atan2, cos, sin, pi, isnan
from PIL import Image
from bosdyn.client import frame_helpers
import io 

class FourWheelsRobot:
    NUMBER_OF_CAMERAS = 1
    NUMBER_OF_MOTORS = 4

    motor_names = [
        "left_front_wheel", "right_front_wheel", "left_back_wheel", "right_back_wheel"
    ]

    camera_names = [
        "frontright_fisheye_image",
        "frontleft_fisheye_image",
        "left_fisheye_image",
        "right_fisheye_image",
        "back_fisheye_image"
    ]

    def __init__(self, runs_in_server=False):
        self.robot = Supervisor()  # Changed from Robot to Supervisor
        self.runs_in_server = runs_in_server
        self.time_step = int(self.robot.getBasicTimeStep())

        # Initialize motors
        self.motors = [self.robot.getDevice(name) for name in self.motor_names]

        # Initialize cameras
        self.cameras = {name: self.robot.getDevice(name) for name in self.camera_names}
        for camera in self.cameras.values():
            camera.enable(2 * self.time_step)

        # initialize other sensors
        self.gps = self.robot.getDevice("gps")
        if self.gps:
            self.gps.enable(self.time_step)

        self.imu = self.robot.getDevice("inertial unit")
        if self.imu:
            self.imu.enable(self.time_step)

    def cast_ray(self, origin, direction, max_distance=5.0):
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Ray direction vector cannot be zero.")
        direction = [d / norm for d in direction]
        target = [origin[i] + direction[i] * max_distance for i in range(3)]
        hit_point = self.robot.rayCast(target, origin)
        if hit_point is None:
            return None
        return {"x": hit_point[0], "y": hit_point[1], "z": hit_point[2]}

    def get_object_locations(self, detections, image_response, target_frame_name=frame_helpers.BODY_FRAME_NAME):
        try:
            img = Image.open(io.BytesIO(image_response.shot.image.data))
            img_width, img_height = img.size
            located_objects = []

            for detection in detections:
                box = detection['box']
                center_px_x = (box[0] + box[2]) / 2
                center_px_y = (box[1] + box[3]) / 2

                focal_length_x = 320
                focal_length_y = 320
                principal_point_x = 320
                principal_point_y = 240

                norm_x = (center_px_x - principal_point_x) / focal_length_x
                norm_y = (center_px_y - principal_point_y) / focal_length_y

                ray_dir = [norm_x, norm_y, 1.0]
                ray_origin = [0.0, 0.1, 0.0]  # Assumed camera position

                hit = self.cast_ray(ray_origin, ray_dir)
                if hit:
                    located_objects.append({
                        'label': detection['label'],
                        'score': detection['score'],
                        'box': detection['box'],
                        'position': hit,
                        'source_camera': "simulation_camera"
                    })

            return located_objects, None

        except Exception as e:
            return None, f"Simulation get_object_locations error: {e}"

    def _wait_for_valid_gps(self):
        if self.runs_in_server:
            return  # skip waiting if simulation already steps externally
        while self.robot.step(self.time_step) != -1:
            position = self.gps.getValues()
            if not any(map(isnan, position)):
                break

    def _wait_for_valid_imu(self):
        if self.runs_in_server:
            return  # skip waiting if simulation already steps externally
        while self.robot.step(self.time_step) != -1:
            orientation = self.imu.getRollPitchYaw()
            if not any(map(isnan, orientation)):
                break

    def get_odometry(self):
        self._wait_for_valid_gps()
        self._wait_for_valid_imu()
        position = self.gps.getValues()
        orientation = self.imu.getRollPitchYaw()  # yaw is last
        
        self.position = {
            "x": position[0], 
            "y": position[1], 
            "z": position[2]
        }
                
        self.orientation = {
            "roll": orientation[0],
            "pitch": orientation[1],
            "yaw": orientation[2]
        }
        
        odometry_data = {
            "position": self.position,
            "orientation": self.orientation
        }
                
        return odometry_data
    

    def step(self):
        return self.robot.step(self.time_step)
    

    def _set_motor_velocity(self, left_speed, right_speed):
        # Left motors: index 0 and 2
        for motor in self.motors:
            motor.setPosition(float('inf'))  # Disable position control
        self.motors[0].setVelocity(left_speed)
        self.motors[2].setVelocity(left_speed)
        # Right motors: index 1 and 3
        self.motors[1].setVelocity(right_speed)
        self.motors[3].setVelocity(right_speed)

    def _normalize_angle(self, angle):
        return (angle + pi) % (2 * pi) - pi
    
    def relative_move(self, delta_x=2, delta_y=1, speed=-6.0):
        # --- Step 1: Get current orientation ---
        odometry = self.get_odometry()
        yaw = odometry["orientation"]["yaw"]  # in radians

        # --- Step 2: Calculate the target direction ---
        relative_angle = atan2(delta_y, delta_x)  # where to turn *relative to robot forward*

        # Target yaw = current yaw + relative_angle
        yaw_target = self._normalize_angle(yaw + relative_angle)
        tolerance = 0.05  # radians

        # --- Step 3: Rotate to the target yaw ---
        start_time = time.time()
        max_rotation_time = 5.0

        while True:
            odometry = self.get_odometry()
            current_yaw = odometry["orientation"]["yaw"]
            yaw_error = self._normalize_angle(yaw_target - current_yaw)

            if abs(yaw_error) < tolerance:
                break

            angular_speed = 2.0 * yaw_error
            angular_speed = max(min(angular_speed, 1.0), -1.0)

            self._set_motor_velocity(angular_speed, -angular_speed)


        self._set_motor_velocity(0, 0)
        time.sleep(0.2)

        # --- Step 4: Drive forward ---
        distance_to_drive = sqrt(delta_x**2 + delta_y**2)

        odometry = self.get_odometry()
        start_pos = odometry["position"]
        start_x = start_pos["x"]
        start_y = start_pos["y"]

        def distance_traveled():
            odometry = self.get_odometry()
            pos = odometry["position"]
            dx = pos["x"] - start_x
            dy = pos["y"] - start_y
            return sqrt(dx**2 + dy**2)

        start_time = time.time()
        max_drive_time = 10.0

        while distance_traveled() < distance_to_drive - 0.05:
            self._set_motor_velocity(speed, speed)


        self._set_motor_velocity(0, 0)

    def turn(self, radians):
        odometry = self.get_odometry()
        yaw = odometry["orientation"]["yaw"]  # in radians

        yaw_target = self._normalize_angle(yaw + radians)
        tolerance = 0.05  # radians

        while True:
            odometry = self.get_odometry()
            current_yaw = odometry["orientation"]["yaw"]
            yaw_error = self._normalize_angle(yaw_target - current_yaw)

            if abs(yaw_error) < tolerance:
                break

            angular_speed = 2.0 * yaw_error
            angular_speed = max(min(angular_speed, 1.0), -1.0)

            self._set_motor_velocity(angular_speed, -angular_speed)

        self._set_motor_velocity(0, 0)

    def sit(self):
        # this robot cannot sit
        pass

    def stand(self):
        # this robot cannot stand
        pass

    def _wait_for_camera_image(self, cam, name, max_attempts=10):
        """Wait until the camera returns a non-null image."""
        for attempt in range(max_attempts):
            self.robot.step(self.time_step)
            raw_image = cam.getImage()
            if raw_image:
                return True
            print(f"Waiting for camera '{name}' to initialize... (Attempt {attempt + 1})")
        print(f"Camera '{name}' did not return a valid image after {max_attempts} attempts.")
        return False


    def take_pictures(self, camera_names):
        """Capture and save images from the specified Webots cameras"""
        image_paths = []

        for name, camera in self.cameras.items():
            if name not in camera_names:
                continue
            self._wait_for_camera_image(camera, name)
            raw_image = camera.getImage()
            if raw_image is None:
                print(f"Image not yet available for camera: {name}")
                continue

            width = camera.getWidth()
            height = camera.getHeight()

            # Convert raw image to numpy array (BGRA)
            img_bgra = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 4))
            
            # Convert BGRA to RGB
            img_rgb = img_bgra[:, :, [2, 1, 0]]

            # Convert to PIL Image
            image = Image.fromarray(img_rgb)

            # Apply rotation based on camera name
            if "frontleft" in name or "frontright" in name:
                pass
                #image = image.rotate(-90, expand=True)
            elif "right" in name:
                image = image.rotate(180, expand=True)

            # Save image
            timestamp = int(time.time())
            filename = f"images/webots_image_{name}_{timestamp}.jpg"
            os.makedirs("images", exist_ok=True)
            image.save(filename, quality=95)
            
            self.last_image_path = filename
            image_paths.append(filename)

        print('image_paths', image_paths)

        return image_paths if image_paths else None


    def relative_move(self, delta_x=2, delta_y=1, speed=-6.0):

        def distance_traveled():
            odometry = self.get_odometry()
            pos = odometry["position"]
            dx = pos["x"] - start_x
            dy = pos["y"] - start_y
            return sqrt(dx**2 + dy**2)

        if delta_x != 0:
            odometry = self.get_odometry()
            start_pos = odometry["position"]
            start_x = start_pos["x"]
            start_y = start_pos["y"]
            
            # Move straight
            current_speed = speed if delta_x > 0 else -speed

            target_distance = abs(delta_x)  # use absolute distance

            while distance_traveled() < (target_distance - 0.05):
                self._set_motor_velocity(current_speed, current_speed)

            self._set_motor_velocity(0, 0)

        if delta_y != 0:
            odometry = self.get_odometry()
            start_yaw = odometry["orientation"]["yaw"]
            start_pos = odometry["position"]
            start_x = start_pos["x"]
            start_y = start_pos["y"]

            while True:
                odometry = self.get_odometry()
                current_yaw = odometry["orientation"]["yaw"]
                yaw_error = self._normalize_angle((start_yaw + (pi/2)) - current_yaw)

                if abs(yaw_error) < 0.05:
                    break

                angular_speed = 2.0 * yaw_error
                angular_speed = max(min(angular_speed, 1.0), -1.0)

                self._set_motor_velocity(angular_speed, -angular_speed)
            
            current_speed = speed if delta_y > 0 else -speed

            target_distance = abs(delta_y)

            while distance_traveled() < (target_distance - 0.05):
                self._set_motor_velocity(current_speed, current_speed)

            self._set_motor_velocity(0, 0)
  
    def get_object_locations(self, detections, image_response, target_frame_name=frame_helpers.BODY_FRAME_NAME):
        try:
            # Open the simulated image
            img = Image.open(io.BytesIO(image_response.shot.image.data))
            img_width, img_height = img.size

            located_objects = []

            for detection in detections:
                box = detection['box']
                center_px_x = (box[0] + box[2]) / 2
                center_px_y = (box[1] + box[3]) / 2

                # Assume focal length and principal point like fake intrinsics
                focal_length_x = 320
                focal_length_y = 320
                principal_point_x = 320
                principal_point_y = 240

                norm_x = (center_px_x - principal_point_x) / focal_length_x
                norm_y = (center_px_y - principal_point_y) / focal_length_y

                depth_estimate = 1.0  # Assume 1 meter in front of robot

                object_x = norm_x * depth_estimate
                object_y = norm_y * depth_estimate
                object_z = depth_estimate

                located_objects.append({
                    'label': detection['label'],
                    'score': detection['score'],
                    'box': detection['box'],
                    'position': {
                        'x': object_x,
                        'y': object_y,
                        'z': object_z
                    },
                    'source_camera': "simulation_camera"
                })

            return located_objects, None

        except Exception as e:
            return None, f"Simulation get_object_locations error: {e}"