import time
import numpy as np
import os
import sys
import io

from spot_controller import SpotController, transform_point_for_rotation

# Bosdyn helpers for camera space conversion / transforms
from bosdyn.client.image import pixel_to_camera_space
from bosdyn.client import frame_helpers

# HTTP session to reuse across calls
import requests

YOLO_API_URL = os.getenv("YOLO_API_URL", "http://localhost:8000")
SESSION = requests.Session()

# Terminal clear command detection
CLEAR_CMD = "cls" if os.name == "nt" else "clear"


def to_unit(vec):
    """Normalize a 3-element NumPy vector. Return None if zero length."""
    n = np.linalg.norm(vec)
    if n == 0:
        return None
    return (vec / n).tolist()


def clear_and_print(vectors):
    """Overwrite terminal with current vectors list."""
    if sys.stdout.isatty():
        print("\033c", end="")  # ANSI clear screen
    else:
        os.system(CLEAR_CMD)

    if vectors:
        print("Person directions (unit vectors in body frame):")
        for i, v in enumerate(vectors, 1):
            print(f"{i}: [{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]")
    else:
        print("No persons detected.")


def rotation_for_camera(src: str) -> int:
    """Return rotation angle (degrees) needed to make image upright for detection."""
    if src in ("frontleft_fisheye_image", "frontright_fisheye_image"):
        return -90  # clockwise
    if src == "right_fisheye_image":
        return 180
    return 0


def process_camera(spot: SpotController, camera_name: str, threshold: float = 0.25):
    """Capture one camera image, run YOLO, return list of body-frame unit vectors to persons."""
    image_resp, err = spot.get_image_and_metadata(camera_name)
    if err or not image_resp:
        return []

    # Original image dimensions
    orig_w = image_resp.shot.image.cols
    orig_h = image_resp.shot.image.rows

    # Build PIL Image & rotate for detection
    from PIL import Image  # local import to avoid heavy import if script unused

    pil_img = Image.open(io.BytesIO(image_resp.shot.image.data))
    rot_angle = rotation_for_camera(camera_name)
    if rot_angle != 0:
        pil_det = pil_img.rotate(rot_angle, expand=True)
    else:
        pil_det = pil_img

    # --- Call YOLO API ---------------------------------------------------
    buf = io.BytesIO()
    pil_det.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    files = {"file": (f"{camera_name}.jpg", buf, "image/jpeg")}
    params = {"threshold": threshold}

    resp = SESSION.post(f"{YOLO_API_URL}/detect/", files=files, params=params, timeout=10)

    if resp.status_code != 200:
        return []

    dets = resp.json().get("detections", [])
    if not dets:
        return []

    # Prepare transform helpers
    t_snapshot = image_resp.shot.transforms_snapshot
    frame_cam = image_resp.shot.frame_name_image_sensor
    cam_to_body = frame_helpers.get_a_tform_b(t_snapshot, frame_cam, frame_helpers.BODY_FRAME_NAME)
    body_to_cam = cam_to_body.inverse()

    vecs_body = []

    # Rotated image dimensions
    rot_w, rot_h = pil_det.size

    for det in dets:
        # YOLO-person API already filters person; treat all
        box = det.get("bbox") or det.get("box")
        if not box or len(box) != 4:
            continue

        cx_rot = (box[0] + box[2]) / 2.0
        cy_rot = (box[1] + box[3]) / 2.0

        # Map rotated coords back to original orientation
        if rot_angle == -90:
            px_orig = cy_rot
            py_orig = orig_h - cx_rot
        elif rot_angle == 180:
            px_orig = orig_w - cx_rot
            py_orig = orig_h - cy_rot
        else:
            # General fallback – uses helper
            px_orig, py_orig = transform_point_for_rotation(
                cx_rot, cy_rot, orig_w, orig_h, rot_w, rot_h, rot_angle)

        # Ray direction in camera frame
        ray_cam = pixel_to_camera_space(image_resp.source, px_orig, py_orig)

        # Rotate into body frame (ignore translation)
        ray_body = body_to_cam.rotation.transform_point(ray_cam[0], ray_cam[1], ray_cam[2])

        vec_body = to_unit(np.array(ray_body))
        if vec_body is not None:
            vecs_body.append(vec_body)

    return vecs_body


def main(loop_delay: float = 0.5):
    spot = SpotController()
    if not spot.connect():
        print("Failed to connect to Spot.")
        return

    cam_sources = [
        "frontleft_fisheye_image",
        "frontright_fisheye_image",
        "left_fisheye_image",
        "right_fisheye_image",
        "back_fisheye_image",
    ]

    try:
        while True:
            vectors = []
            for cam in cam_sources:
                vectors.extend(process_camera(spot, cam))

            clear_and_print(vectors)
            time.sleep(loop_delay)

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        spot.disconnect()


if __name__ == "__main__":
    main() 