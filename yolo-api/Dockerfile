FROM python:3.10-slim

WORKDIR /app

# System-Tools
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

# Anforderungen
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App-Code
COPY app/ app/

# Start der API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
