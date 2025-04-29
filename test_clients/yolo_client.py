import requests
import sys
from PIL import Image, ImageDraw
import io
import time

def test_yolos_api(image_path, api_url="http://134.245.232.230:8002"):
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
        print(f"{i+1}. {detection['label']} ({detection['score']}): {detection['box']}")
    
    # Visualize results
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    for detection in detections:
        box = detection["box"]
        label = detection["label"]
        score = detection["score"]
        
        # Draw box
        draw.rectangle(box, outline="red", width=3)
        
        # Draw label
        text = f"{label} {score}"
        draw.text((box[0], box[1] - 10), text, fill="red")
    
    # Save result
    output_path = "detection_result.jpg"
    img.save(output_path)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    
    test_yolos_api("test.png") 