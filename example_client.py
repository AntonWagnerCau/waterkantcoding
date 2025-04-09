import requests
from mimetypes import guess_type
import time

url = "http://134.245.232.230:8000/caption"
image_path = "spot_image_1744044205.jpg"

with open(image_path, "rb") as f:
    start = time.time()
    response = requests.post(
        url,    
        files = {
        "image_file": (f.name, f, guess_type(image_path)[0] or "image/jpeg")
        }
    )
    end = time.time()
    print(f"Time taken: {end - start} seconds")

if response.status_code == 200:
    print("Caption:", response.json()["caption"])
else:
    print("Error:", response.json())