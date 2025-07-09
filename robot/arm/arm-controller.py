from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
import time

class Arm:
    JOINTS = {
        "j6": {"name": "shoulder_pan.pos", "range": (-115, 115)},
        "j5": {"name": "shoulder_lift.pos", "range": (-100, 100)},
        "j4": {"name": "elbow_flex.pos", "range": (-99, 90)},
        "j3": {"name": "wrist_flex.pos", "range": (-100, 100)},
        "j2": {"name": "wrist_roll.pos", "range": (-160, 160)},
        "j1": {"name": "gripper.pos", "range": (0, 100)},
    }

    INITIAL_POSITION = {
        "shoulder_pan.pos": -10,
        "shoulder_lift.pos": -100,
        "elbow_flex.pos": 99,
        "wrist_flex.pos": 100,
        "wrist_roll.pos": 0,
        "gripper.pos": 0,
    }

    def __init__(self, port="/dev/tty.usbmodem59700731401"):
        self.cfg = SO100FollowerConfig(
            port=port,
            use_degrees=True,
            max_relative_target=None,
        )
        self.robot = SO100Follower(self.cfg)
        self.robot.connect()
        print("Robot connected.")
        self.move_to_initial_position()

    def move_to_initial_position(self):
        print("Moving to initial position...")
        self.robot.send_action(self.INITIAL_POSITION)
        time.sleep(2)
        print("Arm in initial position.")

    def move_joint(self, joint_key, value):
        if joint_key not in self.JOINTS:
            raise ValueError(f"Invalid joint: {joint_key}")
        joint = self.JOINTS[joint_key]
        min_val, max_val = joint["range"]
        value = max(min_val, min(max_val, value))  # Clamp
        self.robot.send_action({joint["name"]: value})
        print(f"Moved {joint['name']} to {value}")

    def interactive_control(self):
        try:
            while True:
                print("\nAvailable joints:")
                for key, info in self.JOINTS.items():
                    r = info["range"]
                    print(f"  {key}: {info['name']} ({r[0]} to {r[1]})")

                joint_input = input("\nEnter joint (j1–j6) or 'q' to quit: ").strip().lower()
                if joint_input == "q":
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
arm = Arm()
arm.interactive_control()