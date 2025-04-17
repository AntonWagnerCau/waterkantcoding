import requests
from mimetypes import guess_type
import time

url = "http://134.245.232.230:8000/caption"
image_path = "images/spot_image_back_fisheye_image_1744204360.jpg"

while True:
    with open(image_path, "rb") as f:
        start = time.time()
        response = requests.post(
            url,    
            files = {
            "file": (f.name, f, guess_type(image_path)[0] or "image/jpeg")
            }
        )
        end = time.time()
        print(f"Time taken: {end - start} seconds")

    if response.status_code == 200:
        print("Caption:", response.json()["caption"])
    else:
        print("Error:", response.json())