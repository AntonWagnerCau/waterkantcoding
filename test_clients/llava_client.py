import requests
import base64
import json
import time
from io import BytesIO
from PIL import Image
from statistics import mean

# --- Configuration ---
API_URL = "http://134.245.232.230:8080/worker_generate_stream" # Worker endpoint 
CONTROLLER_URL = "http://134.245.232.230:8000" # Controller endpoint
IMAGE_URL = "https://llava-vl.github.io/static/images/view.jpg" # Sample image URL
PROMPT = "What is in this image?" # No need for <image> token, it will be added automatically
MODEL_PATH = "liuhaotian/llava-v1.6-mistral-7b" # Model path on Hugging Face
# ---------------------

def load_image(image_file):
    """Load image from file or URL exactly as LLaVA does in their CLI."""
    try:
        if image_file.startswith('http://') or image_file.startswith('https://'):
            print(f"Downloading image from URL: {image_file}")
            response = requests.get(image_file, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
            
        print(f"Successfully loaded image, size: {image.size}, mode: {image.mode}")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def encode_image(image):
    """Encode PIL Image to base64 string."""
    if not image:
        return None
        
    try:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def query_llava_api(image_base64, prompt):
    """Send a request to LLaVA API."""
    headers = {"Content-Type": "application/json"}
    
    # Add image token at the start of the prompt
    prompt_with_image = "<image>\n" + prompt
    
    # LLaVA expects raw base64 without data URI prefix
    payload = {
        "model": MODEL_PATH,
        "prompt": prompt_with_image,
        "images": [image_base64],  # No data URI prefix here, just raw base64
        "temperature": 0.2,
        "max_new_tokens": 1024,
        "stop": "</s>"  # Add a proper stop token
    }

    try:
        print(f"Sending request to: {API_URL}")
        start_time = time.time()
        
        # Stream the response
        response = requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=60)
        response.raise_for_status()
        
        result_text = ""
        print("\nReceiving response:")
        
        # Process streaming response with correct delimiter
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                try:
                    data = json.loads(chunk.decode("utf-8"))
                    result_text = data.get("text", "")
                    
                    # Check for errors
                    if "error_code" in data and data["error_code"] > 0:
                        print(f"\n❌ Error detected: {data}")
                    
                    # Print progressive output
                    print(f"\r{result_text[-80:]}", end="", flush=True)
                except Exception as e:
                    print(f"\nError parsing chunk: {e}")
                    print(f"Raw chunk: {chunk}")
        
        end_time = time.time()
        response_time = end_time - start_time
        print(f"\n\nResponse completed in {response_time:.2f} seconds.")
        return result_text, response_time
    except Exception as e:
        print(f"API request failed: {e}")
        return None, None

def test_images():
    """Test with multiple images from the LLaVA repository."""
    test_images = [
        "https://llava-vl.github.io/static/images/view.jpg",
        "https://raw.githubusercontent.com/haotian-liu/LLaVA/main/images/llava_v1_5_logo.jpg",
        "https://raw.githubusercontent.com/haotian-liu/LLaVA/main/images/math.jpg"
    ]
    
    response_times = []
    
    for img_url in test_images:
        print(f"\n--- Testing with image: {img_url} ---")
        
        # Load and process image using LLaVA's approach
        image = load_image(img_url)
        if not image:
            print(f"Failed to load image from {img_url}")
            continue
            
        # Encode image
        image_base64 = encode_image(image)
        if not image_base64:
            print(f"Failed to encode image from {img_url}")
            continue
            
        # Send API request
        print(f"Sending API request with prompt: {PROMPT}")
        response, response_time = query_llava_api(image_base64, PROMPT)
        
        if response and response_time:
            print("\n--- API Response ---")
            print(response)
            print("------------------")
            response_times.append(response_time)
            return True, response_times  # Return after first successful response
            
    return False, response_times

if __name__ == "__main__":
    print("=== LLaVA API Test Tool ===")
    
    # Test with sample images
    success, response_times = test_images()
    if not success:
        print("\n❌ All image tests failed.")
    else:
        print("\n✅ Image test succeeded!")
        if response_times:
            avg_time = mean(response_times)
            print(f"\nAverage response time: {avg_time:.2f} seconds") 