import requests
import base64
import json
import time
import sys
from io import BytesIO
from PIL import Image

# Configuration
API_URL = "http://localhost:8080/worker_generate_stream"
HEALTH_URL = "http://localhost:8080/health"
IMAGE_URL = "https://llava-vl.github.io/static/images/view.jpg"
PROMPT = """Your task is to give very brief answers to questions asked:
            Should i walk forward or straight to avoid the obstacle and why?
            """

def check_server_health():
    """Check if the server is running and healthy."""
    try:
        response = requests.get(HEALTH_URL, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is healthy: {data}")
            return True
        else:
            print(f"❌ Server returned non-200 status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to server: {e}")
        return False

def load_image_from_url(url):
    """Load image from URL and return as PIL Image."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        print(f"✅ Successfully loaded image: {image.size}x{image.mode}")
        return image
    except Exception as e:
        print(f"❌ Error loading image from URL: {e}")
        return None

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string."""
    try:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        print(f"✅ Successfully encoded image to base64")
        return img_str
    except Exception as e:
        print(f"❌ Error encoding image: {e}")
        return None

def query_tinyllava_api(image_base64, prompt):
    """Send a request to TinyLLaVA API and process the streaming response."""
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "prompt": prompt,
        "images": [image_base64] if image_base64 else None,
        "temperature": 0.2,
        "top_p": 0.7,
        "max_new_tokens": 512
    }

    try:
        print(f"Sending request to: {API_URL}")
        
        # Stream the response
        response = requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=180)
        response.raise_for_status()
        
        result_text = ""
        print("\nReceiving response:")
        
        # Process streaming response with null byte delimiter (\0)
        buffer = b""
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                buffer += chunk
                while b"\0" in buffer:
                    message, buffer = buffer.split(b"\0", 1)
                    if message:
                        try:
                            data = json.loads(message.decode("utf-8"))
                            result_text = data.get("text", "")
                            
                            # Check for errors
                            if "error_code" in data and data["error_code"] > 0:
                                print(f"\n❌ Error detected: {data}")
                            
                            # Print progressive output
                            print(f"\r{result_text[-100:]}", end="", flush=True)
                        except Exception as e:
                            print(f"\nError parsing chunk: {e}")
                            print(f"Raw chunk: {message}")
        
        print("\n\nResponse completed.")
        return result_text
    except Exception as e:
        print(f"❌ API request failed: {e}")
        return None

def main():
    print("=== TinyLLaVA API Test Tool ===")
    
    # Check if server is healthy
    if not check_server_health():
        print("⚠️ Server health check failed. Please make sure the TinyLLaVA API server is running.")
        print("The server may still be initializing. If it's loading a model, it might take a while...")
        
        # Try a few more times with longer waits
        for i in range(3):
            print(f"Retrying health check ({i+1}/3)...")
            time.sleep(20)  # Wait 20 seconds
            if check_server_health():
                break
        else:
            print("❌ Server health check still failing. Exiting.")
            sys.exit(1)
    
    # Load and process image
    print(f"\n--- Loading image from URL: {IMAGE_URL} ---")
    image = load_image_from_url(IMAGE_URL)
    if not image:
        print("❌ Failed to load image. Exiting.")
        sys.exit(1)
    
    # Encode image
    print("\n--- Encoding image to base64 ---")
    image_base64 = encode_image_to_base64(image)
    if not image_base64:
        print("❌ Failed to encode image. Exiting.")
        sys.exit(1)
    
    # Send API request
    print(f"\n--- Sending API request with prompt: {PROMPT} ---")
    response = query_tinyllava_api(image_base64, PROMPT)
    
    if response:
        print("\n--- API Response ---")
        print(response)
        print("------------------")
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed.")

if __name__ == "__main__":
    main() 