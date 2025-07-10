import os
import json
import time
from pathlib import Path

import numpy as np
from ikpy.chain import Chain

from lerobot.robots.so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower import SO100Follower
from lerobot.robots.so101_follower import SO101Follower

# === Load paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path_config = os.path.join(current_dir, "geometry-config", "so101_new_calib.urdf")
robot_config_path = os.path.join(current_dir, "calibration", "robot_one.json")

# === Load robot config JSON ===
with open(robot_config_path, "r") as file:
    raw_config = json.load(file)
print(robot_config_path)

rangeOfMotion ={
    "shoulder_pan":(-100,100),
    "shoulder_lift":(0,160),
    "elbow_flex":(0,-160),
    "wrist_flex":(0,-160),
    "wrist_roll":(0, -160),
    "gripper":(0,100),
}

# === Normalize JOINTS to use keys with '.pos'
JOINTS = {
    f"{name}.pos": {
        **config,
        "name": f"{name}.pos",
        "degree_range": rangeOfMotion.get(name)
    }
    for name, config in raw_config.items()
}



class Arm:
    INITIAL_POSITION  = {
        "shoulder_pan.pos": 0, # -100-100
        "shoulder_lift.pos": -100, # 0- 160
        "elbow_flex.pos": 100,   # 0- -160
        "wrist_flex.pos": 160,  # 0- -160
        "wrist_roll.pos": -160,  # 0- -160
        "gripper.pos": 0,  # 0-100
    }
    JOINTS = JOINTS
    STATE_FILE = "last_position.json"
    def __init__(self, port="COM6", urdf_path=urdf_path_config):
        self.cfg = SO100FollowerConfig(
            port=port,
            use_degrees=True,
            max_relative_target=None,
            calibration_dir=Path("calibration"),
            id="robot_one"
        )
        self.robot = SO100Follower(self.cfg)
        self.robot.connect(calibrate=False)
        print("Robot connected.")

        self.chain = Chain.from_urdf_file(urdf_path)
        last_pos = self.load_last_position()
        self.move_to_position_trapezoidal(self.INITIAL_POSITION, start=last_pos, duration=4.0)
        # self.move_to_position(self.INITIAL_POSITION)

    def move_to_xyz(self, x, y, z):
        print(f"Target XYZ: ({x}, {y}, {z})")

        target_frame = np.eye(4)
        target_frame[:3, 3] = [x, y, z]

        angles = self.chain.inverse_kinematics(target_frame)
        print("IK angles:", angles, type(angles), np.shape(angles))

        joint_action = {}

        # We assume angles[0] is base, skip it if you don't control base joint
        # Order joint keys to match IK output order (usually matches chain.joints)
        joint_keys = list(self.JOINTS.keys())

        if len(angles) - 1 != len(joint_keys):
            print(f"Warning: IK angles ({len(angles) - 1}) and joint keys ({len(joint_keys)}) count mismatch")

        for i, joint_key in enumerate(joint_keys):
            joint = self.JOINTS[joint_key]
            # IK angle at i+1 if base is ignored
            angle_rad = angles[i + 1]
            angle_deg = np.degrees(angle_rad)

            deg_min, deg_max = joint.get("degree_range", (-180, 180))

            # Clamp degree angle within range
            deg_clamped = max(min(angle_deg, deg_max), deg_min)

            # Map degrees to encoder value
            enc_value = int(
                joint["range_min"] +
                (deg_clamped - deg_min) / (deg_max - deg_min) * (joint["range_max"] - joint["range_min"])
            )

            enc_clamped = max(joint["range_min"], min(enc_value, joint["range_max"]))

            joint_action[joint["name"]] = enc_clamped

        print("Calculated joint encoder values:", joint_action)

        self.robot.send_action(joint_action)
        time.sleep(2)

    def move_to_position(self, position):
        print(f"Moving to position...{position}")
        self.robot.send_action(position)
        time.sleep(2)
        self.CURRENT_POSITION = position

    def move_joint(self, joint_key, value):
        if joint_key not in self.JOINTS:
            raise ValueError(f"Invalid joint: {joint_key}")
        joint = self.JOINTS[joint_key]
        clamped = max(joint["range_min"], min(joint["range_max"], value))
        self.robot.send_action({joint["name"]: clamped})
        print(f"Moved {joint['name']} to {clamped}")

    def interactive_control(self):
        try:
            while True:
                print("\nAvailable joints:")
                for key, info in self.JOINTS.items():
                    print(f"  {key}: range {info['range_min']} to {info['range_max']}")

                joint_input = input("\nEnter joint (e.g. shoulder_pan.pos) or 'q' to quit: ").strip().lower()
                if joint_input == "q":
                    self.move_to_position(self.INITIAL_POSITION)
                    break

                if joint_input not in self.JOINTS:
                    print("Invalid joint name.")
                    continue

                pos_input = input(f"Enter position for {joint_input}: ").strip()
                try:
                    value = int(pos_input)
                    self.move_joint(joint_input, value)
                    time.sleep(0.5)
                except ValueError:
                    print("Invalid number.")
        finally:
            self.robot.disconnect()
            print("Robot disconnected.")

    def move_to_position_smooth(self, end, start=None, duration=2.0, steps=50):
        """
        Moves from 'start' to 'end' smoothly using linear interpolation.
        'duration' is total time (in seconds).
        'steps' is how many interpolation points to compute.
        """
        joint_names = self.JOINTS.keys()
        trajectory = []
        start = start if start else self.CURRENT_POSITION
        # start = self.clip_position(start) if start else self.CURRENT_POSITION
        # end = self.clip_position(end)
        
        for step in range(steps + 1):
            t = step / steps
            intermediate = {
                name: int(start[name] + t * (end[name] - start[name]))
                for name in joint_names
            }
            trajectory.append(intermediate)

        interval = duration / steps
        for point in trajectory:
            self.robot.send_action(point)
            self.CURRENT_POSITION = point
            self.save_current_position()
            time.sleep(interval)

    def move_to_position_trapezoidal(self, end, start=None, duration=3.0, steps=100):
        """
        Moves from 'start' to 'end' using a trapezoidal velocity profile.
        Applies per-joint interpolation.
        """
        joint_names = self.JOINTS.keys()
        t = np.linspace(0, 1, steps + 1)
        start = start if start else self.CURRENT_POSITION
        # start = self.clip_position(start) if start else self.CURRENT_POSITION
        # end = self.clip_position(end)
        
        def trapezoidal_profile(t):
            # Simple symmetric trapezoidal velocity profile with t_acc = 0.2
            t_acc = 0.2
            t_dec = 1 - t_acc
            if t < t_acc:
                return 0.5 * (t / t_acc) ** 2
            elif t < t_dec:
                return t_acc / 2 + (t - t_acc)
            else:
                return 1 - 0.5 * ((1 - t) / (1 - t_dec)) ** 2

        trajectory = []
        for step in range(steps + 1):
            s = trapezoidal_profile(t[step])
            point = {
                name: int(start[name] + s * (end[name] - start[name]))
                for name in joint_names
            }
            trajectory.append(point)

        interval = duration / steps
        for point in trajectory:
            self.robot.send_action(point)
            self.CURRENT_POSITION = point
            self.save_current_position()
            time.sleep(interval)
            
    def save_current_position(self):
        with open(self.STATE_FILE, "w") as f:
            json.dump(self.CURRENT_POSITION, f)
        # print("Saved last position.")

    def load_last_position(self):
        if os.path.exists(self.STATE_FILE):
            with open(self.STATE_FILE, "r") as f:
                pos = json.load(f)
            print("Loaded last position from file.")
            return pos
        else:
            print("No saved position found. Using INITIAL_POSITION.")
            return self.INITIAL_POSITION

    def clip_position(self, position):
        """
        Clamps joint values in 'position' to stay within each joint's valid encoder range.
        """
        clipped = {}
        for name, value in position.items():
            if name in self.JOINTS:
                joint = self.JOINTS[name]
                min_val = joint["range_min"]
                max_val = joint["range_max"]
                clipped[name] = max(min(value, max_val), min_val)
            else:
                clipped[name] = value  # fallback, if key is invalid
        return clipped
    
    def move_to_xyz(self, x, y, z, duration=3.0, steps=80, orientation_matrix=None):
        """
        Inverse kinematics solver using ikpy.
        Moves end-effector to (x, y, z) with optional orientation.
        """
        print(f"Target XYZ: ({x:.3f}, {y:.3f}, {z:.3f})")

        # Build 4x4 transformation matrix
        target_frame = np.eye(4)
        target_frame[:3, 3] = [x, y, z]
        if orientation_matrix is not None:
            target_frame[:3, :3] = orientation_matrix  # optional

        # Solve IK
        angles = self.chain.inverse_kinematics(target_frame)
        print("Raw IK angles (radians):", angles)

        # Map to joint names (skip base if unused)
        joint_keys = list(self.JOINTS.keys())
        if len(angles) - 1 != len(joint_keys):
            print(f"⚠️ IK result mismatch: {len(angles)-1} angles vs {len(joint_keys)} joints")

        joint_action = {}
        for i, joint_key in enumerate(joint_keys):
            joint = self.JOINTS[joint_key]
            angle_deg = np.degrees(angles[i + 1])  # skip base
            deg_min, deg_max = joint.get("degree_range", (-180, 180))
            angle_deg = np.clip(angle_deg, deg_min, deg_max)

            # Map to encoder range
            enc_value = int(
                joint["range_min"] +
                (angle_deg - deg_min) / (deg_max - deg_min) * (joint["range_max"] - joint["range_min"])
            )
            enc_value = np.clip(enc_value, joint["range_min"], joint["range_max"])
            joint_action[joint_key] = enc_value

            print("Calculated encoder values:", joint_action)

POINTER_POSITION = {
    "shoulder_pan.pos": 200, # +/-200
    "shoulder_lift.pos": 0, # 0- 160
    "elbow_flex.pos":0,   # 0- -160
    "wrist_flex.pos": 90,  # 0- 160
    "wrist_roll.pos": 0,  # 0- -160
    "gripper.pos": 0,  # 0-100z
}


POSES = [
    {
        "shoulder_pan.pos": 0,
        "shoulder_lift.pos": -100,
        "elbow_flex.pos": 100,
        "wrist_flex.pos": 160,
        "wrist_roll.pos": -160,
        "gripper.pos": 100,
    },
    {
        "shoulder_pan.pos": 30,
        "shoulder_lift.pos": -50,
        "elbow_flex.pos": 80,
        "wrist_flex.pos": 120,
        "wrist_roll.pos": -100,
        "gripper.pos": 32,
    },
    {
        "shoulder_pan.pos": -40,
        "shoulder_lift.pos": -70,
        "elbow_flex.pos": 60,
        "wrist_flex.pos": 90,
        "wrist_roll.pos": -90,
        "gripper.pos": 70,
    },
    {
        "shoulder_pan.pos": 50,
        "shoulder_lift.pos": -120,
        "elbow_flex.pos": 90,
        "wrist_flex.pos": 100,
        "wrist_roll.pos": -30,
        "gripper.pos": 0,
    },
    {
        "shoulder_pan.pos": 0,
        "shoulder_lift.pos": 0,
        "elbow_flex.pos": 0,
        "wrist_flex.pos": 90,
        "wrist_roll.pos": -100,
        "gripper.pos": 50,
    },
    
    {
        "shoulder_pan.pos": 0,
        "shoulder_lift.pos": -90,
        "elbow_flex.pos": 10,
        "wrist_flex.pos": 90,
        "wrist_roll.pos": -20,
        "gripper.pos": 25,
    },
    {
        "shoulder_pan.pos": 0, # +/-200
        "shoulder_lift.pos": 50, # -30+135
        "elbow_flex.pos": -10,   # -10+170
        "wrist_flex.pos": -0,  # +/- 110
        "wrist_roll.pos": 0,  # 0+180
        "gripper.pos": 0,  # 0+100
    }
]

# === Run arm controller ===
if __name__ == "__main__":
    arm = Arm()
    # arm.interactive_control()
    # arm.move_to_position(POINTER_POSITION)
    arm.move_to_position_smooth(POINTER_POSITION, duration=2.0, steps=50)
    # arm.move_to_position_trapezoidal(POINTER_POSITION)
    # arm.move_to_xyz(0.1, 0.1, 0.1)
    for i, pose in enumerate(POSES):
        print(f"\nMoving to Pose {i+1}")
        arm.move_to_position_smooth(pose, duration=2.0, steps=50)
    time.sleep(5)  # Add a short pause after each move if needed

    arm.move_to_position_smooth(arm.INITIAL_POSITION, duration=2.0, steps=50)


