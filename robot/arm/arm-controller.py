import os
import json
import time
from pathlib import Path

import numpy as np
from ikpy.chain import Chain
from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig

# === Load paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path_config = os.path.join(current_dir, "geometry-config", "so101_new_calib.urdf")
robot_config_path = os.path.join(current_dir, "calibration", "robot_one.json")

# === Load robot config JSON ===
with open(robot_config_path, "r") as file:
    raw_config = json.load(file)

# === Normalize JOINTS to use keys with '.pos'
JOINTS = {
    f"{name}.pos": {**config, "name": f"{name}.pos"}
    for name, config in raw_config.items()
}


class Arm:
    JOINTS = JOINTS
    INITIAL_POSITION = {
        "shoulder_pan.pos": 0,
        "shoulder_lift.pos": 0,
        "elbow_flex.pos": 0,
        "wrist_flex.pos": 0,
        "wrist_roll.pos": 0,
        "gripper.pos": 0,
    }


    def __init__(self, port="/dev/tty.usbmodem58760432171", urdf_path=urdf_path_config):
        self.cfg = SO100FollowerConfig(
            port=port,
            use_degrees=True,
            max_relative_target=None,
            calibration_dir=Path("calibration"),
            id="robot_one"
        )
        self.robot = SO100Follower(self.cfg)
        self.robot.connect()
        print("Robot connected.")

        # self.chain = Chain.from_urdf_file(urdf_path)
        self.move_to_position(self.INITIAL_POSITION)

    def move_to_xyz(self, x, y, z):
        print(f"Target XYZ: ({x}, {y}, {z})")

        target_frame = np.eye(4)
        target_frame[:3, 3] = [x, y, z]

        angles = self.chain.inverse_kinematics(target_frame)

        joint_action = {}
        joint_keys = list(self.JOINTS.keys())[::-1]  # j1 to j6
        for i, joint_key in enumerate(joint_keys):
            joint = self.JOINTS[joint_key]
            degrees = np.degrees(angles[i + 1])  # skip base
            clamped = max(joint["range_min"], min(joint["range_max"], degrees))
            joint_action[joint["name"]] = clamped

        print("Calculated joint angles:", joint_action)
        self.robot.send_action(joint_action)
        time.sleep(2)

    def move_to_position(self, position):
        print(f"Moving to position...{position}")
        self.robot.send_action(position)
        # self.robot.disconnect()



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

INITIAL_POSITION = {
    "shoulder_pan.pos": -10,
    "shoulder_lift.pos": 0,
    "elbow_flex.pos": 0,
    "wrist_flex.pos": 0,
    "wrist_roll.pos": 0,
    "gripper.pos": 0,
}
POINTER_POSITION = {
    "shoulder_pan.pos":  0,   # centered
    "shoulder_lift.pos": 100,                # max raise
    "elbow_flex.pos":    -100,                   # fully straight
    "wrist_flex.pos":    -80,                   # fully straight
    "wrist_roll.pos":    0,                # neutral mid-point
    "gripper.pos":       0,                # neutral grip
}

# === Run arm controller ===
if __name__ == "__main__":
    arm = Arm()
    # arm.interactive_control()
    arm.move_to_position(INITIAL_POSITION)
    # arm.move_to_position(POINTER_POSITION)
    # arm.move_to_xyz(0, 0, 10)