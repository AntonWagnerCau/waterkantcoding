"""
SpotAgent - take pictures with a Boston Dynamics Spot robot
"""
import os

from dotenv import load_dotenv
from spot_controller import SpotController

load_dotenv()

# Connect to Spot robot
print("Connecting to Spot robot...")
spot_controller = SpotController()

if os.getenv("SPOT_IP"):
    spot_connected = spot_controller.connect()
    if spot_connected:
        print("Connected to Spot successfully!")
    else:
        print("Failed to connect to Spot. Running in simulation mode.")
else:
    print("No Spot robot configuration found. Running in simulation mode.")
    spot_connected = False

# ATTENTION: test the full connection, make sure no one is in the reach of the robot
# spot_controller.stand()

# get some basic information about Spot's position and orientation
get_odometry = spot_controller.get_odometry()
print(get_odometry)

# now take the pictures
spot_controller.take_pictures()
