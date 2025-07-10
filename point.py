from spot_controller import SpotController
import requests
import os


def send_pictures_to_yolo_api(spot_controller, camera_names=None, yolo_api_url="http://localhost:8000"):
    """
    Take pictures from Spot cameras and send them to local YOLO API for person detection
    
    Args:
        spot_controller: Connected SpotController instance
        camera_names: List of camera names to capture from (default: all fisheye cameras)
        yolo_api_url: URL of the local YOLO API endpoint (default: localhost:8000)
    
    Returns:
        Dictionary with detection results for each image
    """
    
    if camera_names is None:
        camera_names = ["frontleft_fisheye_image", "frontright_fisheye_image", 
                       "back_fisheye_image", "right_fisheye_image", "left_fisheye_image"]
    
    # Take pictures from all specified cameras (in memory)
    print(f"Taking pictures from cameras: {camera_names}")
    images = spot_controller.get_images(camera_names)
    
    if not images:
        print("Failed to capture images")
        return {"error": "Failed to capture images"}
    
    results = {}
    
    # Send each image to the YOLO API
    for i, image in enumerate(images):
        camera_name = camera_names[i]
        print(f"Processing {camera_name}...")
        
        try:
            # Convert PIL image to bytes and send to YOLO API
            import io
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            
            files = {"file": (f"{camera_name}.jpg", img_byte_arr, "image/jpeg")}
            response = requests.post(f"{yolo_api_url}/detect/", files=files)
            
            if response.status_code == 200:
                detections = response.json()["detections"]
                results[camera_name] = {
                    "detections": detections,
                    "person_count": len(detections)
                }
                print(f"  Found {len(detections)} person(s) in {camera_name}")
            else:
                results[camera_name] = {
                    "error": f"API request failed with status {response.status_code}",
                    "response": response.text
                }
                print(f"  Error processing {camera_name}: {response.status_code}")
                
        except Exception as e:
            results[camera_name] = {
                "error": str(e)
            }
            print(f"  Exception processing {camera_name}: {e}")
    
    return results

if __name__ == "__main__":
    spot_controller = SpotController()
    spot_controller.connect()

    # Send pictures to YOLO API (now using in-memory images)
    detection_results = send_pictures_to_yolo_api(spot_controller)
    
    # Print results
    print("\n=== Detection Results ===")
    for camera, result in detection_results.items():
        print(f"\n{camera}:")
        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Persons detected: {result['person_count']}")
            for i, detection in enumerate(result['detections']):
                confidence = detection['confidence']
                bbox = detection['bbox']
                print(f"    Person {i+1}: confidence={confidence:.2f}, bbox={bbox}")

    spot_controller.disconnect()