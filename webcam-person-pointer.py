import math
import time

import cv2
import numpy as np

from robot.arm.armcontroller import Arm
from person_directions import get_persons, process_camera
from robot.camera.cameracotroller import CameraController
from spot_controller import SpotController


import numpy as np
import argparse
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

def list_cameras(max_cams=5):
    print("🔍 Scanning for cameras...")
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera found at ID {i}")
        cap.release()
    else:
        print(f"❌ No camera at ID {i}")
import numpy as np
import math

def compute_camera_intrinsics(width, height, fov_h_deg=70.42, fov_v_deg=43.3):
    fx = (width / 2) / math.tan(math.radians(fov_h_deg) / 2)
    fy = (height / 2) / math.tan(math.radians(fov_v_deg) / 2)
    cx = width / 2
    cy = height / 2
    return fx, fy, cx, cy

def reproject_bbox_center(bbox, width, height):
    """
    bbox = [x1, y1, x2, y2] in pixels
    width, height = image dimensions in px
    Returns a normalized 3D direction vector.
    """
    x1, y1, x2, y2 = bbox
    cx_bb = (x1 + x2) / 2.0
    cy_bb = (y1 + y2) / 2.0

    fx, fy, cx, cy = compute_camera_intrinsics(width, height)

    x_norm = (cx_bb - cx) / fx
    y_norm = (cy_bb - cy) / fy
    vec = np.array([x_norm, y_norm, 1.0], dtype=float)
    vec /= np.linalg.norm(vec)

    return vec

if __name__ == "__main__":
    list_cameras()
    parser = argparse.ArgumentParser(description="YOLO Camera Streamer")
    parser.add_argument("--camera-id", type=int, default=0, help="ID of the USB camera (default: 0)")
    parser.add_argument("--api-url", type=str, default="http://localhost:3001/detect/", help="YOLO detection API endpoint")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between each frame capture")

    args = parser.parse_args()

    controller = CameraController(
        api_url=args.api_url,
        camera_id=args.camera_id,
        interval=args.interval
    )

    controller.start_camera()
    arm = Arm()
    # # Skalierungsfaktor: 50 cm Reichweite
    L = 100
    vec_old = None

    try:
        while True:
            detections = controller.capture_and_send_frame()
            if detections and len(detections) > 0:
                # Get frame size from camera
                width = int(controller.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(controller.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Width: {width} Height: {height}")
                # Reproject all bbox centers to direction vectors
                vectors = [
                    reproject_bbox_center(det["bbox"], width, height)
                    for det in detections
                ]
                if vec_old is not None:
                    vec = findClosestTo(vec_old, vectors)
                else:
                    vec = vectors[0]
                vec = np.abs(vec)
                target = [L * v for v in vec]
                arm.move_to_xyz(*target)
                vec_old = vec
    finally:
        exit(0)





