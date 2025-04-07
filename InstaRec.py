import cv2
import numpy as np
import boto3
import threading
import time
import os
import json
import requests
import uuid
import argparse
from datetime import datetime
from queue import Queue, Empty, Full
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import mss
import mss.tools
import tkinter as tk
from tkinter import simpledialog

# Load environment variables
load_dotenv()

# AWS Configuration
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY") 
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")  
AWS_REGION = "eu-west-1"

# AWS Collection to store face data
COLLECTION_ID = "eyespy-faces"  

# Backend server configuration
DEFAULT_BACKEND_URL = "http://35.180.226.30:8080"
backend_url = os.environ.get("EYESPY_BACKEND_URL", DEFAULT_BACKEND_URL)

# Initialize variables
last_detection_time = 0
detection_throttle = 3.0  # Seconds between detections
processing_enabled = True  # Flag to enable/disable face processing

# Playback settings 
process_every_n = 15  # Process every Nth frame (higher to reduce API calls)
face_display_time = 0.4  # Display face indicator for 0.4 seconds

# Initialize AWS Rekognition client
def get_rekognition_client():
    """Create AWS Rekognition client"""
    try:
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

class ScreenRegionSelector:
    """Simple GUI for selecting a screen region"""
    def __init__(self):
        self.selected_region = None
        
    def get_screen_region(self):
        """Show dialog and get screen region coordinates"""
        # Create a new Tk root window for the dialog
        root = tk.Tk()
        root.title("Select Instagram Live Region")
        root.geometry("500x400")
        root.attributes('-topmost', True)  # Make sure it's on top
        
        tk.Label(root, text="Enter the screen coordinates for Instagram Live feed:",
                 pady=10, font=('Arial', 12, 'bold')).pack()
        
        # Show instructions about positioning
        instructions = (
            "IMPORTANT SETUP INSTRUCTIONS:\n\n"
            "1. Position Instagram Live in your browser BEFORE starting capture\n"
            "2. Make sure the video is clearly visible\n"
            "3. Once you click 'Confirm':\n"
            "   - This window will close completely\n"
            "   - After 2 seconds, the capture window will appear\n"
            "   - The capture window should NOT be in the same area as Instagram"
        )
        tk.Label(root, text=instructions, justify=tk.LEFT, pady=10, 
                 fg="blue", font=('Arial', 10)).pack(padx=20)
        
        frame = tk.Frame(root)
        frame.pack(pady=10)
        
        tk.Label(frame, text="Left:").grid(row=0, column=0, padx=5, pady=5)
        left_entry = tk.Entry(frame, width=5)
        left_entry.grid(row=0, column=1, padx=5, pady=5)
        left_entry.insert(0, "100")
        
        tk.Label(frame, text="Top:").grid(row=0, column=2, padx=5, pady=5)
        top_entry = tk.Entry(frame, width=5)
        top_entry.grid(row=0, column=3, padx=5, pady=5)
        top_entry.insert(0, "100")
        
        tk.Label(frame, text="Width:").grid(row=1, column=0, padx=5, pady=5)
        width_entry = tk.Entry(frame, width=5)
        width_entry.grid(row=1, column=1, padx=5, pady=5)
        width_entry.insert(0, "500")
        
        tk.Label(frame, text="Height:").grid(row=1, column=2, padx=5, pady=5)
        height_entry = tk.Entry(frame, width=5)
        height_entry.grid(row=1, column=3, padx=5, pady=5)
        height_entry.insert(0, "800")
        
        tips_text = ("Tips:\n"
                    "1. Position capture window AWAY from Instagram\n"
                    "2. You can adjust region with arrow keys during capture\n"
                    "3. Press 'q' to quit at any time")
        tk.Label(root, text=tips_text, justify=tk.LEFT, pady=10).pack()
        
        # Use a local function for confirmation to access local variables
        def on_confirm():
            try:
                left = int(left_entry.get())
                top = int(top_entry.get())
                width = int(width_entry.get())
                height = int(height_entry.get())
                
                self.selected_region = {"top": top, "left": left, "width": width, "height": height}
                root.quit()  # This will exit the mainloop but keep the interpreter running
            except ValueError:
                from tkinter import messagebox
                messagebox.showerror("Error", "Please enter valid numbers")
        
        confirm_button = tk.Button(root, text="Confirm", command=on_confirm, 
                                  bg="green", fg="white", font=('Arial', 12, 'bold'))
        confirm_button.pack(pady=20)
        
        # Make sure dialog is modal
        root.protocol("WM_DELETE_WINDOW", root.quit)
        root.mainloop()
        
        # Explicitly destroy the window after mainloop ends
        root.destroy()
        
        # Return the selected region
        return self.selected_region

class ScreenCapture:
    """Screen capture class that captures a specific region of the screen"""
    def __init__(self, region=None):
        self.region = region or {"top": 100, "left": 100, "width": 500, "height": 800}
        self.frame_queue = Queue(maxsize=10)
        self.running = False
        self.fps = 0
        self.last_frame_time = time.time()
        self.frame_count = 0
        
    def start(self):
        """Start capturing from screen region"""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print(f"Started capturing screen region: {self.region}")
        return True
    
    def stop(self):
        """Stop capturing"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        print("Stopped screen capture")
    
    def _capture_loop(self):
        """Continuously capture frames from screen region"""
        with mss.mss() as sct:
            while self.running:
                try:
                    # Capture screenshot of the defined region
                    screenshot = sct.grab(self.region)
                    
                    # Convert to numpy array for OpenCV processing
                    frame = np.array(screenshot)
                    
                    # Convert from BGRA to BGR (remove alpha channel)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    
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
                    
                    # Short sleep to prevent excessive CPU usage
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"Error capturing screen: {e}")
                    time.sleep(0.1)
    
    def get_frame(self):
        """Get the most recent frame"""
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None
    
    def get_fps(self):
        """Get current FPS"""
        return self.fps
    
    def adjust_region(self, dx=0, dy=0, dw=0, dh=0):
        """Adjust the capture region with arrow keys"""
        self.region["left"] += dx
        self.region["top"] += dy
        self.region["width"] += dw
        self.region["height"] += dh
        print(f"Adjusted region: {self.region}")

class FaceDetector:
    """Class to manage face detection with timed face indicators"""
    def __init__(self, face_display_time=1.0):
        self.face_display_time = face_display_time
        self.faces = []  # Current detected faces
        self.face_timestamps = []  # Timestamps for each face detection
        self.detections = []  # Detected face bounding boxes
        self.detection_status = []  # Whether face was new or matched
        self.detection_timestamps = []  # When face was detected
        self.lock = threading.Lock()
        print(f"FaceDetector initialized with display time: {face_display_time} seconds")
        
    def update_faces(self, new_faces):
        """Update detected faces with new timestamp"""
        current_time = time.time()
        with self.lock:
            self.faces = new_faces
            # Reset timestamps for all faces
            self.face_timestamps = [current_time] * len(new_faces)
    
    def get_active_faces(self):
        """Get faces that are still within display time"""
        current_time = time.time()
        active_faces = []
        
        with self.lock:
            # Keep only faces that haven't expired
            for i, (face, timestamp) in enumerate(zip(self.faces, self.face_timestamps)):
                if current_time - timestamp <= self.face_display_time:
                    active_faces.append(face)
        
        return active_faces
        
    def update(self, frame, detected_faces):
        """Update the state with the current frame and detected faces"""
        # Update the faces that are being tracked
        self.update_faces(detected_faces)
        return frame
        
    def register_detection(self, bbox, is_new=True):
        """Register a face detection with timestamp"""
        current_time = time.time()
        with self.lock:
            self.detections.append(bbox)
            self.detection_status.append(is_new)
            self.detection_timestamps.append(current_time)
            
    def draw_indicators(self, frame):
        """Draw indicators for detected faces"""
        # Create a copy of the frame to avoid modifying the original
        result_frame = frame.copy()
        current_time = time.time()
        
        with self.lock:
            # Remove expired detections
            active_indices = []
            for i, timestamp in enumerate(self.detection_timestamps):
                if current_time - timestamp <= self.face_display_time:
                    active_indices.append(i)
            
            # Keep only active detections
            self.detections = [self.detections[i] for i in active_indices if i < len(self.detections)]
            self.detection_status = [self.detection_status[i] for i in active_indices if i < len(self.detection_status)]
            self.detection_timestamps = [self.detection_timestamps[i] for i in active_indices if i < len(self.detection_timestamps)]
            
            # Draw indicators for active detections
            for bbox, is_new in zip(self.detections, self.detection_status):
                left, top, right, bottom = bbox
                
                # Draw different indicators based on whether it's a new face or existing
                if is_new:  # New face
                    # Draw a green circle with "New" text
                    cv2.circle(result_frame, (right-15, top+15), 10, (0, 255, 0), -1)
                    cv2.putText(result_frame, "New", (right-40, top+10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:  # Matched face
                    # Draw a blue circle with "Match" text
                    cv2.circle(result_frame, (right-15, top+15), 10, (255, 0, 0), -1)
                    cv2.putText(result_frame, "Match", (right-50, top+10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return result_frame

def detect_faces_aws(frame):
    """Detect faces using AWS Rekognition with minimal quality filtering"""
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
        
        # Count raw detections for debugging
        total_faces = len(response['FaceDetails'])
        if total_faces > 0:
            print(f"Raw detection: {total_faces} faces found")
            
        # Extract face details with minimal filtering for Instagram
        faces = []
        rejected_faces = {"confidence": 0, "pose": 0, "quality": 0, "landmarks": 0, "size": 0}
        
        for face_detail in response['FaceDetails']:
            # Confidence threshold
            if face_detail['Confidence'] < 80:
                rejected_faces["confidence"] += 1
                continue
            
            # More lenient pose evaluation for Instagram Live
            pose = face_detail['Pose']
            # Instagram users tend to look directly at camera, so be permissive
            if abs(pose['Yaw']) > 60 or abs(pose['Pitch']) > 40:  
                rejected_faces["pose"] += 1
                continue
                
            # Get bounding box
            bbox = face_detail['BoundingBox']
            height, width, _ = frame.shape
            
            # Convert relative coordinates to absolute
            left = int(bbox['Left'] * width)
            top = int(bbox['Top'] * height)
            right = int((bbox['Left'] + bbox['Width']) * width)
            bottom = int((bbox['Top'] + bbox['Height']) * height)
            
            # Minimum size check - smaller for Instagram
            face_height = bottom - top
            if face_height < 60:  # Lower threshold for Instagram
                rejected_faces["size"] += 1
                continue
            
            # If passed all filters, add to faces list
            faces.append({
                'bbox': (left, top, right, bottom),
                'confidence': face_detail['Confidence'],
                'pose': {
                    'yaw': pose['Yaw'],
                    'pitch': pose['Pitch'],
                    'roll': pose['Roll']
                }
            })
        
        return faces
    except Exception as e:
        print(f"Error detecting faces with AWS: {e}")
        return []

def is_new_face_aws(face_img):
    """Check if face is new using AWS Rekognition face search"""
    rekognition = get_rekognition_client()
    if not rekognition:
        print("Could not get Rekognition client")
        return True, None
    
    # Convert image to bytes
    _, img_encoded = cv2.imencode('.jpg', face_img)
    img_bytes = img_encoded.tobytes()
    
    try:
        # Search for face in collection with similarity threshold of 80%
        response = rekognition.search_faces_by_image(
            CollectionId=COLLECTION_ID,
            Image={'Bytes': img_bytes},
            FaceMatchThreshold=80.0, 
            MaxFaces=3  
        )
        
        # If matches found, not a new face
        if response['FaceMatches']:
            best_match = max(response['FaceMatches'], key=lambda x: x['Similarity'])
            matched_face_id = best_match['Face']['FaceId']
            print(f"Found matching face with {best_match['Similarity']:.1f}% similarity (ID: {matched_face_id})")
            return False, matched_face_id
        
        # No matches found, index this face
        index_response = rekognition.index_faces(
            CollectionId=COLLECTION_ID,
            Image={'Bytes': img_bytes},
            MaxFaces=1,
            QualityFilter='AUTO', 
            DetectionAttributes=['DEFAULT']
        )
        
        # Extract the Face ID
        if index_response.get('FaceRecords'):
            face_id = index_response['FaceRecords'][0]['Face']['FaceId']
            print(f"Indexed new face with ID: {face_id}")
            return True, face_id
        else:
            print("Face not indexed - may not meet quality requirements")
            return True, None
    except Exception as e:
        print(f"Error checking face with AWS: {str(e)}")
        return True, None

def is_time_to_detect():
    """Check if enough time has passed since last detection"""
    global last_detection_time
    current_time = time.time()
    if current_time - last_detection_time >= detection_throttle:
        last_detection_time = current_time
        return True
    return False

def upload_to_backend(face_img):
    """Upload a face image directly to the backend server without saving locally"""
    upload_url = f"{backend_url}/api/upload_face"
    
    try:
        print(f"Uploading face to backend at {upload_url}")
        
        # Convert face image to bytes
        _, img_encoded = cv2.imencode('.jpg', face_img)
        img_bytes = img_encoded.tobytes()
        
        # Generate a temporary filename
        temp_filename = f"face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        # Upload directly without saving to disk
        files = {'face': (temp_filename, img_bytes, 'image/jpeg')}
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

def process_face(frame, bbox):
    """Process a detected face - extract, check if new, and upload"""
    try:
        left, top, right, bottom = bbox
        
        # Calculate face dimensions
        face_width = right - left
        face_height = bottom - top
        
        # Add some margin around the face
        margin = min(30, int(face_width * 0.2))
        top = max(0, top - margin)
        bottom = min(frame.shape[0], bottom + margin)
        left = max(0, left - margin)
        right = min(frame.shape[1], right + margin)
        
        # Extract face image
        face_image = frame[top:bottom, left:right]
        
        # Check face size
        if face_image.shape[0] < 100 or face_image.shape[1] < 100:
            print(f"Face too small: {face_image.shape[0]}x{face_image.shape[1]} pixels")
            return None
        
        # Check if enough time has passed since last detection
        if not is_time_to_detect():
            return None
        
        # Check if this is a new face using AWS Rekognition
        is_new, face_id = is_new_face_aws(face_image)
        
        if not is_new:
            print(f"Face matched existing face with ID: {face_id}")
            return {'matched': True, 'face_id': face_id}
        
        if not face_id:
            print("No face ID returned - face may not be suitable for indexing")
            return None
        
        # Upload face to backend (without saving locally)
        thread = threading.Thread(
            target=upload_to_backend,
            args=(face_image,),
            daemon=True
        )
        thread.start()
        
        return {'matched': False, 'face_id': face_id}
    except Exception as e:
        print(f"Error processing face: {e}")
        return None

def main(server_url=None):
    """Main function"""
    global backend_url, processing_enabled
    
    print("\n==== Instagram Live Face Recognition ====")
    
    # Check if AWS credentials are configured
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        print("\nERROR: AWS credentials not configured!")
        print("Please set AWS_ACCESS_KEY and AWS_SECRET_KEY environment variables")
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
        print(f"Warning: Backend server is not responding at {backend_url}")
        choice = input("Continue anyway? (y/n): ")
        if choice.lower() != 'y':
            return
    
    print("\nSETUP INSTRUCTIONS:")
    print("1. Position Instagram Live in your browser")
    print("2. Note the screen coordinates (you'll enter them in the next step)")
    print("3. The capture window should NOT overlap with Instagram")
    input("Press Enter when ready to select region...")
    
    # Get screen region from user
    selector = ScreenRegionSelector()
    region = selector.get_screen_region()
    
    if not region:
        print("No region selected. Using default region.")
        region = {"top": 100, "left": 100, "width": 500, "height": 800}
    
    print("\nRegion selected. Starting capture in 2 seconds...")
    print(f"Selected region: Top={region['top']}, Left={region['left']}, Width={region['width']}, Height={region['height']}")
    
    # Add a delay to ensure the selector window is completely closed
    time.sleep(2)
    
    # Create screen capture
    capture = ScreenCapture(region)
    capture.start()
    
    # Create monitoring window and position it away from the capture region
    cv2.namedWindow("Instagram Face Recognition", cv2.WINDOW_NORMAL)
    
    # Position the window away from the Instagram region to avoid recursive capture
    monitor_width = 800
    monitor_height = 600
    
    # Try to position the window on the opposite side of the screen from the capture region
    if region["left"] > 800:
        # Instagram is on the right, put our window on the left
        cv2.moveWindow("Instagram Face Recognition", 50, 50)
    else:
        # Instagram is on the left, put our window on the right
        cv2.moveWindow("Instagram Face Recognition", max(region["left"] + region["width"] + 50, 100), 50)
    
    cv2.resizeWindow("Instagram Face Recognition", monitor_width, monitor_height)
    
    # Create face detector
    face_detector = FaceDetector(face_display_time=face_display_time)
    
    # Variables for detection
    frame_counter = 0
    face_detection_count = 0
    matched_faces_count = 0
    
    # Face detection thread variables
    face_detection_thread_active = False
    face_frame = None
    detected_faces = []
    
    def face_detection_worker():
        nonlocal face_frame, face_detection_count, face_detection_thread_active, detected_faces, matched_faces_count
        
        print("Face detection worker started")
        processing_count = 0
        
        while processing_enabled and face_detection_thread_active:
            if face_frame is not None:
                local_frame = face_frame.copy()
                face_frame = None  # Clear the frame so we don't process it again
                processing_count += 1
                
                try:
                    # Detect faces with AWS Rekognition
                    faces = detect_faces_aws(local_frame)
                    
                    if faces:
                        # Update face detector with new faces for display
                        face_detector.update_faces([face['bbox'] for face in faces])
                        
                        # Store all detected faces for display
                        detected_faces = faces
                        
                        # Process each detected face
                        for face in faces:
                            bbox = face['bbox']
                            result = process_face(local_frame, bbox)
                            
                            if result:
                                if result.get('matched', False):
                                    matched_faces_count += 1
                                else:
                                    face_detection_count += 1
                                    # Register detection to show indicator
                                    face_detector.register_detection(bbox, not result.get('matched', False))
                
                except Exception as e:
                    print(f"Error in face detection thread: {e}")
                
                # Print status every 10 processed frames
                if processing_count % 10 == 0:
                    print(f"Processed {processing_count} frames, found {face_detection_count} new faces, matched {matched_faces_count}")
            
            # Short sleep to prevent CPU overuse
            time.sleep(0.01)
        
        print("Face detection thread stopped")
    
    # Start face detection thread
    face_detection_thread_active = True
    detection_thread = threading.Thread(target=face_detection_worker, daemon=True)
    detection_thread.start()
    
    # Print usage instructions
    print("\nINSTRUCTIONS:")
    print("- Press 'p' to pause/resume face processing")
    print("- Arrow keys to adjust capture region")
    print("- Press 'q' to quit")
    print(f"- Processing 1 frame every {process_every_n} frames to reduce API costs")
    print(f"- Face detection throttled to once every {detection_throttle} seconds")
    print(f"- Detected faces will be uploaded to {backend_url}")
    
    try:
        # Main monitoring loop
        running = True
        
        while running:
            # Get frame from screen capture
            frame = capture.get_frame()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Increment frame counter
            frame_counter += 1
                
            # Process only selected frames to reduce AWS API calls
            if processing_enabled and frame_counter % process_every_n == 0:
                # Send frame to detection thread
                if face_frame is None:  # Only update if previous frame was processed
                    face_frame = frame.copy()
            
            # Get active faces
            active_faces = face_detector.get_active_faces()
            
            # Prepare display frame
            display_frame = frame.copy()
            
            # Let the face detector draw recognition indicators
            display_frame = face_detector.draw_indicators(display_frame)
            
            # Draw rectangles around active faces
            for (left, top, right, bottom) in active_faces:
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Add status text
            if processing_enabled:
                status_text = f"Processing: ON (1 frame every {process_every_n})"
                status_color = (0, 255, 0)  # Green
            else:
                status_text = "Processing: PAUSED (press 'p' to resume)"
                status_color = (0, 0, 255)  # Red
            
            cv2.putText(display_frame, f"Instagram Live Face Recognition", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {capture.get_fps():.1f} | Faces: {len(active_faces)}", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"New faces: {face_detection_count} | Matched: {matched_faces_count}", 
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, status_text, 
                      (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
            # Show frame
            cv2.imshow("Instagram Face Recognition", display_frame)
            
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
            # Arrow keys to adjust region
            elif key == 0x51:  # Left arrow
                capture.adjust_region(dx=-10)
            elif key == 0x52:  # Up arrow
                capture.adjust_region(dy=-10)
            elif key == 0x53:  # Right arrow
                capture.adjust_region(dx=10)
            elif key == 0x54:  # Down arrow
                capture.adjust_region(dy=10)
            # Shift + arrow keys to adjust size
            elif key == ord('['):  # Decrease width
                capture.adjust_region(dw=-10)
            elif key == ord(']'):  # Increase width
                capture.adjust_region(dw=10)
            elif key == ord('-'):  # Decrease height
                capture.adjust_region(dh=-10)
            elif key == ord('='):  # Increase height
                capture.adjust_region(dh=10)
    
    except KeyboardInterrupt:
        print("Monitoring stopped by user")
    except Exception as e:
        print(f"Error during monitoring: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        face_detection_thread_active = False
        detection_thread.join(timeout=1.0)
        capture.stop()
        cv2.destroyAllWindows()
        print("\nMonitoring stopped")
        
        # Print statistics
        print(f"- New faces detected: {face_detection_count}")
        print(f"- Matched faces: {matched_faces_count}")
        print(f"- All faces uploaded to: {backend_url}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Instagram Live Face Recognition')
    parser.add_argument('--server', default=None, help='Backend server URL (default: http://35.180.226.30:8080)')
    
    args = parser.parse_args()
    main(server_url=args.server)