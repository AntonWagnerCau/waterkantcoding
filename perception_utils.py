"""
Perception utilities for SpotAgent.
Includes YOLO-based object detection and 3D localization.
"""
import os
import io
import json
import requests
from bosdyn.client import ray_cast
from bosdyn.client import frame_helpers
from bosdyn.api import geometry_pb2
from bosdyn.client.image import build_image_request
import bosdyn.api.image_pb2 as image_pb2

class ObjectDetector:
    """Handles object detection and 3D localization for Spot robot."""
    
    def __init__(self, image_client=None):
        """Initialize the detector with optional image client.
        
        Args:
            image_client: A bosdyn.client.image.ImageClient instance
        """
        self.image_client = image_client
        self.api_url = os.getenv("YOLO_API_URL", "http://134.245.232.230:8002")
    
    def get_image_and_metadata(self, source_name="frontleft_fisheye_image"):
        """Capture image and the necessary SDK metadata for 3D projection.
        
        Args:
            source_name: The camera source to use
            
        Returns:
            A tuple of (image_response, error_str)
        """
        if not self.image_client:
            return None, "Image client not initialized"
            
        image_request = build_image_request(
            source_name, 
            quality_percent=100,
            image_format=image_pb2.Image.FORMAT_JPEG,
            pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8
        )
        
        image_responses = self.image_client.get_image([image_request])
        if not image_responses:
            return None, "No image received"
            
        return image_responses[0], None
    
    def detect_objects_in_image(self, image_response):
        """Send image data to external YOLO endpoint.
        
        Args:
            image_response: The bosdyn ImageResponse object
            
        Returns:
            A tuple of (detections, error_str)
        """
        if not image_response or not image_response.shot.image.data:
            return None, "Invalid image_response"

        try:
            image_bytes = io.BytesIO(image_response.shot.image.data)
            files = {'file': ('image.raw', image_bytes)}
            params = {'threshold': 0.5}

            response = requests.post(
                f"{self.api_url}/detect", 
                files=files, 
                params=params, 
                timeout=10
            )
            response.raise_for_status()
            detections = response.json().get("detections", [])
            return detections, None
            
        except requests.exceptions.RequestException as e:
            return None, f"Error calling YOLO API: {e}"
        except json.JSONDecodeError as e:
            return None, f"Error parsing JSON response from YOLO API: {e}"
        except Exception as e:
            return None, f"Unexpected error calling YOLO API: {e}"
    
    def get_object_locations(self, detections, image_response, target_frame_name=frame_helpers.BODY_FRAME_NAME):
        """Convert 2D pixel coordinates to 3D points in a chosen robot frame.
        
        Args:
            detections: List of detection objects from YOLO API
            image_response: The bosdyn ImageResponse object
            target_frame_name: Frame to express positions in (default: body frame)
            
        Returns:
            A tuple of (located_objects, error_str)
        """
        if not detections or not image_response:
            return None, "Missing detections or image_response"

        located_objects = []
        # Extract metadata needed for ray_cast
        transforms_snapshot = image_response.shot.transforms_snapshot
        camera_intrinsics = image_response.source.pinhole
        frame_name_image_sensor = image_response.shot.frame_name_image_sensor

        for detection in detections:
            # Calculate center pixel of the bounding box
            box = detection['box']
            center_px_x = (box[0] + box[2]) / 2
            center_px_y = (box[1] + box[3]) / 2
            pixel_vec = geometry_pb2.Vec2(x=center_px_x, y=center_px_y)

            try:
                # Use the SDK's ray_cast utility
                hit_position_in_image_frame, distance = ray_cast.ray_cast_pixel(
                    pixel_vec, transforms_snapshot, frame_name_image_sensor, camera_intrinsics)

                if distance < 0:  # Ray cast miss
                    continue  # Skip this detection

                # Transform the hit point to the desired target frame
                frame_tform_target = frame_helpers.get_a_tform_b(
                    transforms_snapshot, target_frame_name, frame_name_image_sensor)
                    
                if frame_tform_target is None:
                    continue  # Skip if transform fails

                point_in_target_frame = frame_tform_target.transform_point(
                    x=hit_position_in_image_frame.x,
                    y=hit_position_in_image_frame.y,
                    z=hit_position_in_image_frame.z)

                # Store result
                located_objects.append({
                    'label': detection['label'],
                    'score': detection['score'],
                    'position': {
                        'x': point_in_target_frame[0],
                        'y': point_in_target_frame[1],
                        'z': point_in_target_frame[2]
                    }
                })
            except Exception as e:
                print(f"Error processing detection {detection.get('label', 'N/A')}: {e}")
                continue  # Continue with next detection

        return located_objects, None
        
    def locate_objects_in_view(self, image_source="frontleft_fisheye_image", target_frame=frame_helpers.BODY_FRAME_NAME):
        """Combine object detection steps: capture image, detect objects, project to 3D.
        
        Args:
            image_source: The camera source to use
            target_frame: Frame to express positions in (default: body frame)
            
        Returns:
            A tuple of (located_objects, error_str)
        """
        # 1. Get Image and Metadata
        image_response, error = self.get_image_and_metadata(image_source)
        if error: 
            return None, error

        # 2. Detect Objects via YOLO
        detections, error = self.detect_objects_in_image(image_response)
        if error: 
            return None, error
        if not detections: 
            return [], None  # No detections is not an error

        # 3. Project Detections to 3D
        located_objects, error = self.get_object_locations(detections, image_response, target_frame)
        if error: 
            return None, error

        return located_objects, None 