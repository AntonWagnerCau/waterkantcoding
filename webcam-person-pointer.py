import argparse
import time

import cv2

from robot.arm.armcontroller import Arm
from robot.camera.cameracotroller import CameraController


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
import numpy as np
import time

def move_in_circle(arm, radius=10, center_x=0, center_z=0, y=0, num_points=30, delay=1):
    """
    Moves the robot arm in a circular path in the X-Z plane.

    Parameters:
        arm: Arm instance with move_to_xyz(x, y, z) method
        radius (float): Radius of the circle
        center_x (float): X-coordinate of the circle's center
        center_z (float): Z-coordinate of the circle's center
        y (float): Fixed Y-level
        num_points (int): Number of points along the circle
        delay (float): Delay in seconds between moves
    """
    print("⭮ Starting circular motion...")

    # Generate circular positions
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    positions = [
        (center_x + radius * np.cos(theta), y, center_z + radius * np.sin(theta))
        for theta in angles
    ]

    # Move through the circular path
    for idx, (x, y, z) in enumerate(positions):
        print(f"➡️ Moving to point {idx+1}: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        arm.move_to_xyz(x, y, z)
        time.sleep(delay)

    print("✅ Circular motion complete.")
import time
def move_in_square_xy(arm, side_length=10, center_x=0, center_y=0, z=0, delay=5):
    """
    Moves the robot arm in a square path in the X–Y plane.

    Parameters:
        arm: Arm instance with move_to_xyz(x, y, z) method
        side_length (float): Length of one side of the square
        center_x (float): X-coordinate of the square's center
        center_y (float): Y-coordinate of the square's center
        z (float): Fixed Z level
        delay (float): Delay in seconds between moves
    """
    print("⬛ Starting square motion in X–Y plane...")

    half_side = side_length / 2.0

    # Define corners of the square in X–Y, Z is fixed
    corners = [
        (center_x - half_side, center_y - half_side, z),
        (center_x + half_side, center_y - half_side, z),
        (center_x + half_side, center_y + half_side, z),
        (center_x - half_side, center_y + half_side, z),
    ]

    # Optionally loop back to the start to close the square
    corners.append(corners[0])

    for idx, (x, y, z) in enumerate(corners):
        print(f"➡️ Moving to corner {idx+1}: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        arm.move_to_xyz(x, y, z)
        time.sleep(delay)

    print("✅ Square motion in X–Y plane complete.")
def move_in_square(arm, side_length=20, center_x=0, center_z=0, y=0, delay=5):
    """
    Moves the robot arm in a square path in the X-Z plane.

    Parameters:
        arm: Arm instance with move_to_xyz(x, y, z) method
        side_length (float): Length of one side of the square
        center_x (float): X-coordinate of the square's center
        center_z (float): Z-coordinate of the square's center
        y (float): Fixed Y-level
        delay (float): Delay in seconds between moves
    """
    print("⬛ Starting square motion...")

    half_side = side_length / 2.0

    # Define 4 corners of the square (clockwise or counterclockwise)
    corners = [
        (center_x - half_side, y, center_z - half_side),
        (center_x + half_side, y, center_z - half_side),
        (center_x + half_side, y, center_z + half_side),
        (center_x - half_side, y, center_z + half_side),
    ]

    # Optionally loop back to the starting point
    corners.append(corners[0])  # to close the square

    for idx, (x, y, z) in enumerate(corners):
        print(f"➡️ Moving to corner {idx+1}: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        arm.move_to_xyz(x, y, z)
        time.sleep(delay)

    print("✅ Square motion complete.")
if __name__ == "__main__":
    list_cameras()
    parser = argparse.ArgumentParser(description="YOLO Camera Streamer")
    parser.add_argument("--camera-id", type=int, default=0, help="ID of the USB camera (default: 0)")
    parser.add_argument("--api-url", type=str, default="http://localhost:3001/detect/", help="YOLO detection API endpoint")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between each frame capture")

    args = parser.parse_args()
    #
    # controller = CameraController(
    #     api_url=args.api_url,
    #     camera_id=args.camera_id,
    #     interval=args.interval
    # )

    # controller.start_camera()
    arm = Arm()
    # # Skalierungsfaktor: 50 cm Reichweite
    L = 100
    vec_old = None
    vec =[0,0,0]
    vec1 =[0,0,0]
    vx, vy, vz = vec
    zrange = [-20,30]
    xrange = [-30,30]
    arm.move_to_xyz(*vec)
    time.sleep(2)
    # move_in_circle(arm)
    move_in_square_xy(arm)

    # arm.move_to_xyz(*vec)

    # try:
    #     while True:
    #         controller.capture_and_send_frame()
    #         detections = controller.getDetections()
    #         if detections and len(detections) > 0:
    #             # Get frame size from camera
    #             width = int(controller.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #             height = int(controller.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #             print(f"Width: {width} Height: {height}")
    #             # Reproject all bbox centers to direction vectors
    #             vectors = [
    #                 reproject_bbox_center(det["bbox"], width, height)
    #                 for det in detections
    #             ]
    #             if vec_old is not None:
    #                 vec = findClosestTo(vec_old, vectors)
    #             else:
    #                 vec = vectors[0]
    #             vec = np.abs(vec)
    #             target = [L * v for v in vec]
    #             print(target)
    #             arm.move_to_xyz(*target)
    #             vec_old = vec
    #             time.sleep(3)
    # finally:
    #     exit(0)
    # #




