from fastapi import FastAPI, File, UploadFile
from app.detect import detect_persons

app = FastAPI()

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    detections = detect_persons(image_bytes)
    return {"detections": detections}
