import cv2
import numpy as np
import boto3
import threading
import time
import os
import pickle
import json
import requests
import uuid
from datetime import datetime
from queue import Queue, Empty, Full
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create directory to save face screenshots
save_dir = os.path.join(script_dir, "detected_faces")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created detected faces directory: {save_dir}")

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY") 
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")  
AWS_REGION = "eu-west-1" 

# AWS Collection to store face data - You can change this name if desired
COLLECTION_ID = "eyespy-faces"  

# Backend server configuration
DEFAULT_BACKEND_URL = "http://localhost:8080"
backend_url = os.environ.get("EYESPY_BACKEND_URL", DEFAULT_BACKEND_URL)

# Initialize variables
known_face_ids = set()
last_detection_time = 0
detection_throttle = 1.0  # Seconds between detections
processing_enabled = True  # Flag to enable/disable face processing

# Initialize AWS Rekognition client with direct credentials
def get_rekognition_client():
    """Create AWS Rekognition client"""
    try:
        # Initialize with explicit credentials
        rekognition = boto3.client(
            'rekognition',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        return rekognition
    except Exception as e:
        print(f"Error initializing AWS Rekognition: {e}")
        return None

# Load previously seen face IDs if the file exists
face_ids_file = os.path.join(save_dir, "known_face_ids.pkl")
if os.path.exists(face_ids_file):
    try:
        with open(face_ids_file, 'rb') as f:
            known_face_ids = pickle.load(f)
        print(f"Loaded {len(known_face_ids)} previously detected face IDs")
    except Exception as e:
        print(f"Error loading known face IDs: {e}")
        known_face_ids = set()

def ensure_collection_exists():
    """Ensure the AWS Rekognition Collection exists"""
    rekognition = get_rekognition_client()
    if not rekognition:
        return False
    
    try:
        # Check if collection exists
        response = rekognition.list_collections()
        
        if COLLECTION_ID not in response['CollectionIds']:
            # Create collection
            print(f"Creating new AWS Rekognition collection: {COLLECTION_ID}")
            rekognition.create_collection(CollectionId=COLLECTION_ID)
        else:
            print(f"Using existing collection: {COLLECTION_ID}")
        
        return True
    except Exception as e:
        print(f"Error with AWS collection: {e}")
        return False

class WebcamCapture:
    """Webcam capture class using direct OpenCV access"""
    def __init__(self, camera_id=0, width=640, height=480):
        self.camera_id = camera_id
        self.width = width 
        self.height = height
        self.frame_queue = Queue(maxsize=2)
        self.running = False
        self.thread = None
        self.fps = 0
        self.last_frame_time = time.time()
        self.frame_count = 0
        
    def start(self):
        """Start capturing from webcam"""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print(f"Started capturing from webcam {self.camera_id}")
        return True
    
    def stop(self):
        """Stop capturing"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        print("Stopped webcam capture")
    
    def _capture_loop(self):
        """Continuously capture frames from webcam"""
        # Initialize webcam
        camera = cv2.VideoCapture(self.camera_id)
        
        # Set resolution
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Set camera buffer size to 1 frame
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Webcam initialized at resolution: {actual_width}x{actual_height}")
        
        while self.running:
            try:
                # Capture frame
                ret, frame = camera.read()
                
                if not ret:
                    print("Error reading from webcam")
                    time.sleep(0.1)
                    continue
                
                # Update queue
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                    
                    # Update FPS calculation
                    self.frame_count += 1
                    if self.frame_count >= 30:
                        now = time.time()
                        self.fps = self.frame_count / (now - self.last_frame_time)
                        self.frame_count = 0
                        self.last_frame_time = now
                except Empty:
                    pass
            except Exception as e:
                print(f"Error capturing frame: {e}")
                time.sleep(0.1)
        
        # Release camera when done
        camera.release()
    
    def get_frame(self):
        """Get the most recent frame"""
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None
    
    def get_fps(self):
        """Get current FPS"""
        return self.fps

def detect_faces_aws(frame):
    """Detect faces using AWS Rekognition"""
    rekognition = get_rekognition_client()
    if not rekognition:
        return []
    
    # Convert frame to bytes
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()
    
    try:
        # Detect faces with AWS
        response = rekognition.detect_faces(
            Image={'Bytes': img_bytes},
            Attributes=['DEFAULT']
        )
        
        # Extract face details
        faces = []
        for face_detail in response['FaceDetails']:
            if face_detail['Confidence'] < 90:  # Skip low confidence detections
                continue
                
            # Get bounding box
            bbox = face_detail['BoundingBox']
            height, width, _ = frame.shape
            
            # Convert relative coordinates to absolute
            left = int(bbox['Left'] * width)
            top = int(bbox['Top'] * height)
            right = int((bbox['Left'] + bbox['Width']) * width)
            bottom = int((bbox['Top'] + bbox['Height']) * height)
            
            faces.append({
                'bbox': (left, top, right, bottom),
                'confidence': face_detail['Confidence']
            })
        
        return faces
    except Exception as e:
        print(f"Error detecting faces with AWS: {e}")
        return []

def is_new_face_aws(face_img):
    """Check if face is new using AWS Rekognition face search"""
    rekognition = get_rekognition_client()
    if not rekognition:
        return True, None
    
    # Convert image to bytes
    _, img_encoded = cv2.imencode('.jpg', face_img)
    img_bytes = img_encoded.tobytes()
    
    try:
        # Search for face in collection
        response = rekognition.search_faces_by_image(
            CollectionId=COLLECTION_ID,
            Image={'Bytes': img_bytes},
            FaceMatchThreshold=80.0,
            MaxFaces=1
        )
        
        # If matches found, not a new face
        if response['FaceMatches']:
            match = response['FaceMatches'][0]
            print(f"Found matching face with {match['Similarity']:.1f}% similarity")
            return False, None
        
        # If no matches, index this new face
        index_response = rekognition.index_faces(
            CollectionId=COLLECTION_ID,
            Image={'Bytes': img_bytes},
            MaxFaces=1,
            QualityFilter='AUTO',
            DetectionAttributes=['DEFAULT']
        )
        
        # Extract the Face ID
        if index_response['FaceRecords']:
            face_id = index_response['FaceRecords'][0]['Face']['FaceId']
            print(f"Indexed new face with ID: {face_id}")
            return True, face_id
        
        return True, None
    except ClientError as e:
        # Handle specific error for no faces detected
        if "InvalidParameterException" in str(e):
            print("No suitable face found in the image")
            return False, None
        print(f"AWS error: {e}")
        return True, None  # Assume new face if error
    except Exception as e:
        print(f"Error checking face with AWS: {e}")
        return True, None  # Assume new face if error

def is_time_to_detect():
    """Check if enough time has passed since last detection"""
    global last_detection_time
    current_time = time.time()
    if current_time - last_detection_time >= detection_throttle:
        last_detection_time = current_time
        return True
    return False

def upload_to_backend(file_path):
    """Upload a face image to the backend server"""
    upload_url = f"{backend_url}/api/upload_face"
    
    try:
        print(f"Uploading face to backend at {upload_url}")
        with open(file_path, 'rb') as f:
            files = {'face': (os.path.basename(file_path), f, 'image/jpeg')}
            response = requests.post(upload_url, files=files, timeout=10)
        
        if response.status_code == 200:
            print(f"Face uploaded successfully: {response.json()}")
            return True
        else:
            print(f"Face upload failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Error uploading face to backend: {e}")
        return False

def check_backend_health():
    """Check if backend server is accessible"""
    try:
        health_url = f"{backend_url}/api/health"
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            print(f"Backend server is healthy at {backend_url}")
            return True
        else:
            print(f"Backend server returned unexpected status: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to backend server at {backend_url}: {e}")
        return False

def save_face(frame, bbox):
    """Save a detected face if it's new and upload to backend"""
    global known_face_ids
    
    # Extract face from bounding box
    try:
        left, top, right, bottom = bbox
        
        # Add margin to face crop
        margin = 30
        top = max(0, top - margin)
        bottom = min(frame.shape[0], bottom + margin)
        left = max(0, left - margin)
        right = min(frame.shape[1], right + margin)
        
        # Extract face image
        face_image = frame[top:bottom, left:right]
        
        # Check if too small or invalid
        if face_image.shape[0] < 50 or face_image.shape[1] < 50:
            return None
        
        # Check if this is a new face
        is_new, face_id = is_new_face_aws(face_image)
        
        if not is_new or not face_id:
            return None
        
        # Only save if enough time has passed
        if not is_time_to_detect():
            return None
        
        # If we have a valid face ID, add to known faces
        known_face_ids.add(face_id)
        
        # Generate filename with timestamp and face ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/face_{timestamp}_{face_id[:8]}.jpg"
        
        # Save face image
        cv2.imwrite(filename, face_image)
        print(f"New face saved: {filename}")
        
        # Save updated known face IDs
        with open(face_ids_file, 'wb') as f:
            pickle.dump(known_face_ids, f)
        
        # Upload to backend
        thread = threading.Thread(
            target=upload_to_backend,
            args=(filename,),
            daemon=True
        )
        thread.start()
        
        return filename
    except Exception as e:
        print(f"Error saving face: {e}")
        return None

def list_camera_devices():
    """List available camera devices"""
    available_cameras = []
    for i in range(10):  # Check first 10 indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Camera available
            available_cameras.append(i)
            cap.release()
    return available_cameras

def select_camera():
    """Select a camera from available devices"""
    available_cameras = list_camera_devices()
    
    if not available_cameras:
        print("No cameras detected!")
        return None
    
    print("\nDetected cameras:")
    for i, cam_id in enumerate(available_cameras):
        print(f"{i+1}. Camera {cam_id}")
    
    if len(available_cameras) == 1:
        print(f"Only one camera detected. Using camera {available_cameras[0]}")
        return available_cameras[0]
    
    while True:
        try:
            choice = input("\nSelect camera (1-{}): ".format(len(available_cameras)))
            idx = int(choice) - 1
            if 0 <= idx < len(available_cameras):
                return available_cameras[idx]
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\nSelection cancelled")
            return None

def main(server_url=None):
    """Main function"""
    global backend_url, processing_enabled
    
    print("\n==== AWS Rekognition Webcam Monitor ====")
    
    # Check if AWS credentials are configured
    if AWS_ACCESS_KEY == "YOUR_ACCESS_KEY_HERE" or AWS_SECRET_KEY == "YOUR_SECRET_KEY_HERE":
        print("\nERROR: AWS credentials not configured!")
        print("Please edit this script and replace the placeholder values for:")
        print("- AWS_ACCESS_KEY")
        print("- AWS_SECRET_KEY")
        print("- AWS_REGION (if needed)")
        return
    
    # Check AWS connection
    rekognition = get_rekognition_client()
    if not rekognition:
        print("ERROR: Could not connect to AWS Rekognition.")
        print("Please check your credentials and internet connection.")
        return
    
    # Create collection
    if not ensure_collection_exists():
        print("ERROR: Could not create or access AWS Rekognition collection.")
        return
    
    # Set backend URL if provided
    if server_url:
        backend_url = server_url
        print(f"Using custom backend URL: {backend_url}")
    
    # Check backend connectivity
    if not check_backend_health():
        print("Warning: Backend server is not responding. Face processing will be local only.")
        print(f"Make sure the backend server is running at {backend_url}")
    
    # Select camera
    camera_id = select_camera()
    if camera_id is None:
        print("No camera selected. Exiting.")
        return
    
    # Create webcam capture
    capture = WebcamCapture(camera_id=camera_id)
    capture.start()
    
    # Create monitoring window
    cv2.namedWindow("Face Monitoring", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Monitoring", 800, 600)
    
    # Variables for processing
    process_every_n = 30  # Process every 30th frame to reduce costs
    frame_counter = 0
    current_faces = []
    face_detection_count = 0
    
    # Print cost information
    print("\nCOST INFORMATION:")
    print("- AWS Rekognition: $1 per 1,000 face operations")
    print(f"- Processing 1 frame every {process_every_n} frames to reduce costs")
    print("- Press 'p' to pause processing completely")
    
    try:
        # Main monitoring loop
        running = True
        while running:
            # Get frame
            frame = capture.get_frame()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Increment frame counter
            frame_counter += 1
                
            # Process only selected frames to reduce AWS API calls
            if processing_enabled and frame_counter % process_every_n == 0:
                try:
                    # Use AWS Rekognition to detect faces
                    faces = detect_faces_aws(frame)
                    
                    # Update current faces
                    current_faces = [face['bbox'] for face in faces]
                    
                    # Process each detected face
                    for face in faces:
                        bbox = face['bbox']
                        # Save if it's a new face
                        if save_face(frame, bbox):
                            face_detection_count += 1
                except Exception as e:
                    print(f"Error processing frame: {e}")
            
            # Draw rectangles around faces
            display_frame = frame.copy()
            for (left, top, right, bottom) in current_faces:
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Get processing status text
            if processing_enabled:
                status_text = f"Processing every {process_every_n} frames"
                status_color = (0, 255, 255)  # Yellow
            else:
                status_text = "Processing PAUSED (press 'p' to resume)"
                status_color = (0, 0, 255)  # Red
            
            # Add status text
            cv2.putText(display_frame, f"AWS Rekognition (Camera {camera_id})", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {capture.get_fps():.1f} | Faces: {len(current_faces)}", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Unique faces: {len(known_face_ids)}", 
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, status_text, 
                      (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_frame, "Press 'q' to quit, 'p' to pause/resume processing", 
                      (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                
            # Show frame
            cv2.imshow("Face Monitoring", display_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested")
                running = False
            elif key == ord('p'):
                # Toggle processing
                processing_enabled = not processing_enabled
                status = "RESUMED" if processing_enabled else "PAUSED"
                print(f"Face processing {status}")
    
    except KeyboardInterrupt:
        print("Monitoring stopped by user")
    except Exception as e:
        print(f"Error during monitoring: {e}")
    finally:
        # Clean up
        capture.stop()
        cv2.destroyAllWindows()
        print("\nMonitoring stopped")
        print(f"- Faces saved to: {os.path.abspath(save_dir)}")
        print(f"- {len(known_face_ids)} unique faces detected")
        print(f"- {face_detection_count} total face detections")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='AWS Rekognition Webcam Monitor')
    parser.add_argument('--server', default=None, help='Backend server URL (default: http://localhost:8000)')
    
    args = parser.parse_args()
    main(server_url=args.server)