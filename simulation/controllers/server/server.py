from controller import Robot
import sys
import os



sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..", "..")))
from spot_controller import SpotController
from spot_agent import SpotAgent
import time

if __name__ == "__main__":
    print("Starting setup")
    controller = SpotController(isSimulation=True)
    print('Controller initialized', controller)

    if controller.simulation:
        controller.simulation.start_in_thread()  # <-- Start simulation loop in background
        print('Simulation started in background thread.')

    spot_agent = SpotAgent(controller)
    spot_agent.start()
    print('Agent initialized', spot_agent)
    print("[Main] Server running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Main] Shutdown requested. Exiting...")


#######################
# nc localhost 65432
#######################