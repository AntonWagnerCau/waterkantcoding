import math
import time

import numpy as np

from robot.arm.armcontroller import Arm
from person_directions import get_persons, process_camera
from spot_controller import SpotController


import numpy as np

def findClosestTo(vec_old, vectors):
    """
    Gibt den Vektor aus 'vectors' zurück, der am ähnlichsten zu 'vec_old' ist.
    Alle Vektoren müssen normiert sein (Einheitsvektoren).

    :param vec_old: np.array, normierter Referenzvektor
    :param vectors: list of list[float] oder list of np.array, alle normiert
    :return: np.array, der ähnlichste Vektor
    """
    if not vectors:
        return None

    # Alle Vektoren in np.array-Form bringen
    vectors_np = [np.array(v) for v in vectors]

    # Cosinus-Ähnlichkeiten (Skalarprodukt für normierte Vektoren)
    similarities = [np.dot(vec_old, v) for v in vectors_np]

    # Index des ähnlichsten Vektors
    best_index = np.argmax(similarities)

    return vectors_np[best_index]


if __name__ == "__main__":
    arm = Arm()
    spot = SpotController()
    # # Skalierungsfaktor: 50 cm Reichweite
    L = 1000
    vec_old = None
    if not spot.connect("image"):
        print("Failed to connect to Spot.")
    try:
        while True:
            vectors = []
            for cam in ["frontleft_fisheye_image","frontright_fisheye_image"]:
                vectors.extend(process_camera(spot, cam))
            if vectors.__len__() > 0:
                if vec_old:
                    vec = findClosestTo(vec_old,vectors)
                else:
                    vec = vectors[0]
                target = [L * v for v in vec]
                # Einheitsvektor (normiert)
                vx, vy, vz = vec  # vec = vectors[0]
                # Zielposition = 50 cm in diese Richtung
                L = 1000
                x = L * vx
                y = L * vy
                z = 10
                # Optional: Greiferöffnung
                open_val = 0.5
                # Bewegung ausführen
                arm.move_to_xyz(*target)
                vec_old = vec  # z.B. vorheriger "vec"
    finally:
        spot.disconnect()





