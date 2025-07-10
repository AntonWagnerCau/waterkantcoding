import requests
import sys
from PIL import Image, ImageDraw
import io
import time

def test_yolos_api(image_path, api_url="http://127.0.0.1:8000"):
    """Test the YOLOS API with a local image"""
    
    # Send image to API
    with open(image_path, 'rb') as f:
        files = {'file': f}
        params = {'threshold': 0.1}  # Lower threshold to detect more objects
        
        print(f"Sending request to {api_url}/detect")
        start = time.time()
        response = requests.post(f"{api_url}/detect", files=files, params=params)
        end = time.time()
        print(f"Time taken: {end - start} seconds")
    
    # Check response
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return
    
    # Get results
    results = response.json()
    detections = results.get("detections", [])
    
    print(f"Found {len(detections)} objects:")

    for i, detection in enumerate(detections):
        print(f"{i+1}. ({detection['confidence']}): {detection['bbox']}")
    
    # Visualize results
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    for detection in detections:
        bbox = detection["bbox"]

        # Draw box
        draw.rectangle(bbox, outline="red", width=3)
        
        # Draw label
        #text = f"{label} {score}"
    #    draw.text((box[0], box[1] - 10), text, fill="red")
    
    # Save result
    output_path = "detection_result.jpg"
    img.save(output_path)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    
    test_yolos_api("../images/spot_image_frontright_fisheye_image_1752055740.jpg")