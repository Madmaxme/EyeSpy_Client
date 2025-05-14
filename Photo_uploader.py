import os
import requests
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Backend server configuration
DEFAULT_BACKEND_URL = 'http://localhost:8080' #http://localhost:8080
backend_url = os.environ.get("EYESPY_BACKEND_URL", DEFAULT_BACKEND_URL)

def upload_photo(file_path):
    """Upload a photo to the backend server"""
    upload_url = f"{backend_url}/api/upload_face"
    
    try:
        print(f"Uploading photo to backend at {upload_url}")
        with open(file_path, 'rb') as f:
            files = {'face': (os.path.basename(file_path), f, 'image/jpeg')}
            response = requests.post(upload_url, files=files, timeout=10)
        
        if response.status_code == 200:
            print(f"Photo uploaded successfully: {response.json()}")
            return True
        else:
            print(f"Photo upload failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Error uploading photo to backend: {e}")
        return False

def main():
    # Parse command line arguments
    global backend_url
    parser = argparse.ArgumentParser(description='Simple photo uploader for EyeSpy')
    parser.add_argument('photo_path', nargs='?', help='Path to the photo file to upload')
    parser.add_argument('--url', help=f'Backend URL (default: {backend_url})')
    args = parser.parse_args()
    
    # Check if photo_path was provided
    if not args.photo_path:
        print("Error: Please provide a path to a photo file")
        print("Example: python simple_photo_uploader.py /path/to/your/photo.jpg")
        return
    
    # Update backend URL if provided
    if args.url:
        backend_url = args.url
        print(f"Using custom backend URL: {backend_url}")
    
    # Verify file exists
    if not os.path.exists(args.photo_path):
        print(f"Error: File not found: {args.photo_path}")
        return
    
    # Upload the photo
    upload_photo(args.photo_path)

if __name__ == "__main__":
    main()
