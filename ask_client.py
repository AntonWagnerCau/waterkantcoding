import requests
import os

API_ENDPOINT = "http://134.245.232.230:8000/ask"

def ask_question_minimal(question: str, image_path: str = None, image_url: str = None):
    """Sends question and image (path OR url) to the API."""
    data = {"question": question}
    files = None
    opened_file = None

    try:
        if image_path:
            if not os.path.exists(image_path): return {"error": f"File not found: {image_path}"}
            opened_file = open(image_path, 'rb')
            files = {'image_file': (os.path.basename(image_path), opened_file)}
        elif image_url:
            data['image_url'] = image_url
        else:
            return {"error": "Provide image_path or image_url"}

        response = requests.post(API_ENDPOINT, data=data, files=files, timeout=60)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx/5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}", "status_code": getattr(e.response, 'status_code', None)}
    finally:
        if opened_file:
            opened_file.close() # Ensure file is closed

if __name__ == "__main__":
    result = ask_question_minimal("To get to the door, i need to walk to my", image_path="images/spot_image_back_fisheye_image_1744187365.jpg")

    print(result)