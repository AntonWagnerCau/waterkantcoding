import cv2
import requests
import time
import argparse


class CameraController:
    def __init__(self, api_url="http://localhost:3001/detect/", camera_id=0, interval=1.0):
        self.api_url = api_url
        self.camera_id = camera_id
        self.interval = interval
        self.cap = None
        self.running = False
        self.detections = []

    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Could not open camera ID {self.camera_id}")
        self.running = True
        print(f"🚀 Streaming from camera ID {self.camera_id} to {self.api_url}...")

    def draw_boxes(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            confidence = det["confidence"]
            label = f"Person: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def capture_and_send_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Failed to capture frame.")
            return []

        _, img_encoded = cv2.imencode('.jpg', frame)
        image_bytes = img_encoded.tobytes()

        files = {'file': ('frame.jpg', image_bytes, 'image/jpeg')}
        try:
            response = requests.post(self.api_url, files=files)
            response.raise_for_status()
            detections = response.json().get("detections", [])
            print(f"✅ Detected {len(detections)} person(s)")
            self.draw_boxes(frame, detections)
            self.detections = detections
        except requests.RequestException as e:
            print("⚠️ Request failed:", e)
            detections = []

        # Always show the camera feed, even if request failed
        cv2.imshow("Camera Feed (Press 'q' to quit)", frame)

    def getDetections(self):
        return self.detections
    def run(self):
        self.start_camera()
        try:
            while self.running:
                self.capture_and_send_frame()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user.")
        finally:
            self.release()

    def release(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("📷 Camera released.")


