import time
import numpy as np
import os
import sys
import io
import tkinter as tk
import math

from PIL import Image, ImageTk, ImageDraw, ImageFont

from spot_controller import SpotController, transform_point_for_rotation
from dotenv import load_dotenv

# Bosdyn helpers for camera space conversion / transforms
from bosdyn.client.image import pixel_to_camera_space
from bosdyn.client import frame_helpers

# HTTP session to reuse across calls
import requests

YOLO_API_URL = os.getenv("YOLO_API_URL", "http://localhost:8085")
SESSION = requests.Session()

# Load environment variables
load_dotenv()

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
            print(f"{i}: [{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}, degrees: {v[3]:+.0f}]")
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

    # fill overlapping with black to prevent YOLO from detecting
    img = Image.open(buf).convert("RGB")
    draw = ImageDraw.Draw(img)
    if camera_name=='frontleft_fisheye_image':
        draw.rectangle([0, 0, 74, img.height - 1], fill=(0, 0, 0))
    elif camera_name=='left_fisheye_image':
        draw.rectangle([600, 0, 639, img.height - 1], fill=(0, 0, 0))
    elif camera_name=='right_fisheye_image':
        draw.rectangle([0, 0, 40, img.height -1], fill=(0, 0, 0))

    buf_new = io.BytesIO()
    img.save(buf_new, format="JPEG", quality=95)
    buf_new.seek(0)

    files = {"file": (f"{camera_name}.jpg", buf_new, "image/jpeg")}
    params = {"threshold": threshold}

    current_images[camera_name] = buf_new

    resp = SESSION.post(f"{YOLO_API_URL}/detect/", files=files, params=params, timeout=10)

    if resp.status_code != 200:
        return []

    dets = resp.json().get("detections", [])
    if not dets:
        return []

    # Draw boxes on images
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    draw = ImageDraw.Draw(image)

    for det in dets:
        x_min, y_min, x_max, y_max = det["bbox"]
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

    img_with_boxes = io.BytesIO()
    image.save(img_with_boxes, format="JPEG")
    img_with_boxes.seek(0)
    current_images[camera_name] = img_with_boxes

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
            # calculate and display degrees
            degree = vec_to_angle_deg(vec_body[0], vec_body[1])

            vec_body.append(degree)
            vecs_body.append(vec_body)

    return vecs_body

class GuiApp:
    def __init__(self, root, cam_sources, current_images):
        self.root = root
        self.cam_sources = cam_sources
        self.current_images = current_images
        self.labels = {}

        special_frame = tk.Frame(root, padx=5, pady=5, relief=tk.RIDGE, borderwidth=2, bg="lightgray")
        special_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5)

        image = Image.open("assets/spot_top_view.jpg")
        tk_image = ImageTk.PhotoImage(image)

        image_label = tk.Label(special_frame, image=tk_image, bg="lightgray")
        image_label.image = tk_image
        self.image_label = image_label
        image_label.pack()

        special_label = tk.Label(special_frame, text="Bot", bg="lightgray")
        special_label.pack()

        allowed_positions = [(0, 0), (0, 2), (1, 0), (1, 2), (2, 1)]

        for cam, (row, col) in zip(cam_sources, allowed_positions):
            frame = tk.Frame(root, padx=5, pady=5, relief=tk.RIDGE, borderwidth=1)
            frame.grid(row=row, column=col, padx=5, pady=5)

            label = tk.Label(frame, text=cam)
            label.pack()

            image_label = tk.Label(frame)
            image_label.pack()

            self.labels[cam] = image_label

    def update_gui_loop(self, vectors):

        center_x, center_y = 300, 300
        length = 250

        image = Image.open("assets/spot_top_view.jpg")
        draw = ImageDraw.Draw(image)

        for vec in vectors:

            # we need to rotate 90 degrees counterclockwise
            rotated_x = -vec[1]
            rotated_y = vec[0]

            dx = rotated_x * length
            dy = -rotated_y * length

            draw.line(
                [(center_x, center_y), (center_x + dx, center_y + dy)],
                fill="red",
                width=2
            )

            textoffset = 15
            if vec[3] < 90 and vec[3] > 270:
                textoffset = -15

            text_x = center_x + dx - textoffset
            text_y = center_y + dy - textoffset

            font = ImageFont.load_default()

            draw.text((text_x, text_y), str(round(vec[3])) + '°', fill="black", font=font)

        tk_image = ImageTk.PhotoImage(image)

        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image

        for cam in self.cam_sources:
            img_data = self.current_images[cam].getvalue()
            if img_data:
                try:
                    pil_image = Image.open(io.BytesIO(img_data))
                    pil_image = pil_image.resize((640, 480))
                    tk_image = ImageTk.PhotoImage(pil_image)
                    self.labels[cam].configure(image=tk_image)
                    self.labels[cam].image = tk_image
                except Exception as e:
                    print(f"Error displaying {cam}: {e}")
        self.root.update_idletasks()
        self.root.update()

def vec_to_angle_deg(x, y):
    angle_rad = math.atan2(y, x)
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def main(cam_sources, loop_delay: float = 3):
    spot = SpotController()
    if not spot.connect('image'):
        print("Failed to connect to Spot.")
        return

    try:
        while True:
            vectors = []
            for cam in cam_sources:
                vectors.extend(process_camera(spot, cam))

            clear_and_print(vectors)

            # update GUI
            guiApp.update_gui_loop(vectors)

            time.sleep(loop_delay)

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        spot.disconnect()

cam_sources = [
        "frontright_fisheye_image",
        "frontleft_fisheye_image",
        "left_fisheye_image",
        "right_fisheye_image",
        "back_fisheye_image",
    ]

current_images = {name: io.BytesIO() for name in cam_sources}

# GUI erstellen
root = tk.Tk()
guiApp = GuiApp(root, cam_sources, current_images)

if __name__ == "__main__":
    main(cam_sources)