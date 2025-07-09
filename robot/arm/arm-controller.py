import os
import json
import time
from pathlib import Path

import numpy as np
from ikpy.chain import Chain

from lerobot.robots.so100_follower import SO100FollowerConfig
from robot.arm.lerobot.robots.so100_follower import SO100Follower
from robot.arm.lerobot.robots.so101_follower import SO101Follower

# === Load paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path_config = os.path.join(current_dir, "geometry-config", "so101_new_calib.urdf")
robot_config_path = os.path.join(current_dir, "calibration", "robot_one.json")

# === Load robot config JSON ===
with open(robot_config_path, "r") as file:
    raw_config = json.load(file)

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
        "shoulder_lift.pos": 0, # 0- 160
        "elbow_flex.pos": 0,   # 0- -160
        "wrist_flex.pos": 0,  # 0- -160
        "wrist_roll.pos": -160,  # 0- -160
        "gripper.pos": 0,  # 0-100
    }
    JOINTS = JOINTS
    def __init__(self, port="/dev/tty.usbmodem58760432171", urdf_path=urdf_path_config):
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
        self.move_to_position(self.INITIAL_POSITION)

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
        time.sleep(4)

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



POINTER_POSITION = {
    "shoulder_pan.pos": 0, # -100-100
    "shoulder_lift.pos": 80, # 0- 160
    "elbow_flex.pos": -80,   # 0- -160
    "wrist_flex.pos": -100,  # 0- -160
    "wrist_roll.pos": -160,  # 0- -160
    "gripper.pos": 0,  # 0-100
}

# === Run arm controller ===
if __name__ == "__main__":
    arm = Arm()
    # arm.interactive_control()
    arm.move_to_position(POINTER_POSITION)
    # arm.move_to_xyz(0.1, 0.1, 0.1)

















