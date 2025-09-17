# Copyright 1996-2024 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0
import sys
import os

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..", "..")))
from spot_controller import SpotController

class SpotTest:
    def __init__(self):
        self.spot = SpotController(isSimulation=True)


    def run(self):
        print("Start run", self.spot)
        self.spot.walk_forward(distance_meters=1)
        self.spot.take_pictures()
        self.spot.locate_objects_in_view()
        #self.spot.take_pictures()
        #self.spot.analyze_images()
        self.spot.walk_backward(distance_meters=1)
        self.spot.turn(90)
        self.spot.turn(-45)
        #self.spot.take_pictures()

print("Before run")
if __name__ == "__main__":
    controller = SpotTest()
    controller.run()
