import os

import requests

# === Load paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path_config = os.path.join(current_dir, "geometry-config", "so101_new_calib.urdf")
robot_config_path = os.path.join(current_dir, "calibration", "robot_one.json")
jestsonIp = "192.168.10.235"


class Arm:

    def move_to_xyz(self, x, y, z, rx=0, ry=0, rz=0, open_val=0,
                    max_trials=10, position_tolerance=0.03, orientation_tolerance=0.2, robot_id=0):
        url = f"http://{jestsonIp}:3000/move/absolute?robot_id={robot_id}"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        payload = {
            "x": x,
            "y": y,
            "z": z,
            "rx": rx,
            "ry": ry,
            "rz": rz,
            "open": open_val,
            "max_trials": max_trials,
            "position_tolerance": position_tolerance,
            "orientation_tolerance": orientation_tolerance
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            print("Move command successful:", response.json())
        except requests.exceptions.RequestException as e:
            print("Failed to send move command:", e)



















