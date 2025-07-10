import time

from robot.arm.armcontroller import Arm
from person_directions import get_persons, process_camera
from spot_controller import SpotController

if __name__ == "__main__":
    arm = Arm()
    spot = SpotController()
    # # Skalierungsfaktor: 50 cm Reichweite
    L = 3
    if not spot.connect():
        print("Failed to connect to Spot.")


    try:
        while True:
            vectors = []
            for cam in ["frontright_fisheye_image"]:
                vectors.extend(process_camera(spot, cam))
            time.sleep(3)

            if vectors.__len__() > 0:
                print(vectors)
                vec = vectors[0]
                target = [L * v for v in vec]
                print(target)
            arm.move_to_xyz(10,20,20)
    finally:
        spot.disconnect()





