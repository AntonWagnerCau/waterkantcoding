from robots.so100_follower import SO100Follower, SO100FollowerConfig
import time
from ikpy.chain import Chain
import numpy as np

cfg = SO100FollowerConfig(
    port="/dev/tty.usbmodem59700731401",
    use_degrees=True,
    max_relative_target=None,
    #max_relative_target=20.0,
)

JOINT_LIMITS = {
    "shoulder_pan.pos":   (-115, 115),
    "shoulder_lift.pos":  (-100, 100),
    "elbow_flex.pos":     (-99, 90),
    "wrist_flex.pos":     (-100, 100),
    "wrist_roll.pos":     (-160, 160),
    "gripper.pos":        (0, 100),
}

# Load IK chain from URDF
robot_chain = Chain.from_urdf_file("src/lerobot/so101_new_calib.urdf")
#robot_chain = Chain.from_urdf_file("src/lerobot/so101_old_calib.urdf")

def move_to_xyz(x: float, y: float, z: float):
    """
    Use inverse kinematics to move the gripper to (x, y, z) in meters.
    Joint values are clamped using JOINT_LIMITS.
    """
    target_frame = np.eye(4)
    target_frame[:3, 3] = [x, y, z]

    # IK mit Ziel-Frame (4x4 Matrix)
    #ik_result = robot_chain.inverse_kinematics(target_frame, target_position=None)
    ik_result = robot_chain.inverse_kinematics([x, y, z])

    action = {
        "shoulder_pan.pos":   ik_result[1] * 180 / np.pi,
        "shoulder_lift.pos":  ik_result[2] * 180 / np.pi,
        "elbow_flex.pos":     ik_result[3] * 180 / np.pi,
        "wrist_flex.pos":     ik_result[4] * 180 / np.pi,
        "wrist_roll.pos":     ik_result[5] * 180 / np.pi,
        "gripper.pos":        50.0  # optional default
    }

    # Clamp to joint limits
    for joint, (min_val, max_val) in JOINT_LIMITS.items():
        if joint in action:
            orig = action[joint]
            action[joint] = max(min(orig, max_val), min_val)
            if action[joint] != orig:
                print(f"Clamped {joint}: {orig} -> {action[joint]}")

    robot.send_action(action)
    time.sleep(2)

robot = SO100Follower(cfg)
robot.connect()
robot.send_action({"gripper.pos": 0 }) 
move_to_xyz(0.3, 0.2, 0.25)
time.sleep(0.5)  

#robot.disconnect()


