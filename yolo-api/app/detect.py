from ultralytics import YOLO
from PIL import Image
import io

# Modell laden (nur einmal)
model = YOLO("yolov8n.pt")  # n, s, m, l, x je nach Wunsch

def detect_persons(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(image)

    # Nur Personen (class_id == 0)
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # person
                detections.append({
                    "class_id": cls_id,
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })
    return detections
