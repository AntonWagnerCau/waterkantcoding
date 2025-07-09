# 🧠 YOLOv8 Person Detection API – Install_Readme.md

This API uses [YOLOv8](https://github.com/ultralytics/ultralytics) for detecting **persons** in images and provides a simple REST interface – all inside a Docker container.

---

## ⚙️ Requirements

- [Docker](https://www.docker.com/) installed
- Optional: `curl` or a REST client like Postman

---

## 🚧 Build the Docker Image (one-time setup)

```bash
docker build -t yolo-api .
```

🔹 This builds a Docker image with YOLOv8 and FastAPI.  
🔹 The model `yolov8n.pt` will be downloaded automatically on first run.

---

## ▶️ Start the API

```bash
docker run -p 8000:8000 yolo-api
```

🔹 The API will be available at: [http://localhost:8000](http://localhost:8000)

---

## 📤 Upload an Image for Detection

Send an image via POST request to `/detect/`:

```bash
curl -X POST http://localhost:8000/detect/ \
  -H "accept: application/json" \
  -F "file=@images\spot_image_frontright_fisheye_image_1752055982.jpg"
```

📦 Example response:

```json
{
  "detections": [
    {
      "class_id": 0,
      "confidence": 0.91,
      "bbox": [34.5, 12.3, 180.7, 305.4]
    }
  ]
}
```

Only objects of class **"person"** (COCO class ID 0) will be detected.

---

## 📑 API Endpoint

| Method | Path       | Description                          |
|--------|------------|--------------------------------------|
| POST   | `/detect/` | Upload image, return detected people |

---

## 🧼 Stop the Container (optional)

```bash
docker ps          # Get container ID
docker stop <ID>   # Stop it
```

---

## 💡 Notes

- The default model `yolov8n.pt` is small and fast. For better accuracy, you can replace it with `yolov8s.pt`, `yolov8m.pt`, or `yolov8l.pt` (update `detect.py` manually).
- Detection is limited to persons only (class_id == 0).