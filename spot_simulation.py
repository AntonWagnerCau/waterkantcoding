import sys
import os
import threading
from bosdyn.client import frame_helpers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.controllers.robots.four_wheels_robot import FourWheelsRobot


class SpotSimulation:

    last_image_path = None
    
    def __init__(self):
        self.robot = FourWheelsRobot()

    def step(self):
        if self.robot.step(self.time_step) == -1:
            sys.exit(0)
    
    def run(self):
        print("Starting simulation loop...")
        while self.robot.step() != -1:
            # (Optional) robot updates
            pass
        print("Simulation loop ended.")
    
    def start_in_thread(self):
        """Start the simulation loop in a background thread."""
        self.robot.runs_in_server = True
        threading.Thread(target=self.run, daemon=True).start()

    def relative_move(self, delta_x, delta_y):
        return self.robot.relative_move(delta_x, delta_y)

    def turn(self, radians):
        return self.robot.turn(radians)

    def sit(self):
        return self.robot.sit()

    def stand(self):
        return self.robot.stand()
    
    def get_odometry(self):
        return self.robot.get_odometry()

    def take_pictures(self, camera_names=[]):
        last_image_paths = self.robot.take_pictures(camera_names)
        self.last_image_path = last_image_paths[0]
        return last_image_paths

    def get_object_locations(self, detections, image_response, target_frame_name=frame_helpers.BODY_FRAME_NAME):
        return self.robot.get_object_locations(detections, image_response, frame_helpers.BODY_FRAME_NAME)


   

    