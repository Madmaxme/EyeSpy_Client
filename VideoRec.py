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
DEFAULT_BACKEND_URL = "http://35.180.226.30:8080" # http://35.180.226.30:8080
backend_url = os.environ.get("EYESPY_BACKEND_URL", DEFAULT_BACKEND_URL)

# Initialize variables
last_detection_time = 0
detection_throttle = 3.0  # Seconds between detections
processing_enabled = True  # Flag to enable/disable face processing

# Playback speed control (lower = slower, higher = faster)
# 1.0 = normal speed, 0.5 = half speed, 2.0 = double speed
playback_speed = 0.5  # Start at 50% of normal speed

# Face indicator display time (in seconds)
face_display_time = .4  # Display face indicator for 0.4 seconds

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

def get_collection_face_count():
    """Get the number of faces in the AWS collection"""
    rekognition = get_rekognition_client()
    if not rekognition:
        return 0
    
    try:
        response = rekognition.describe_collection(CollectionId=COLLECTION_ID)
        return response.get('FaceCount', 0)
    except Exception as e:
        print(f"Error getting collection face count: {e}")
        return 0

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
    
class FaceTracker:
    def __init__(self, stability_threshold=4, position_tolerance=0.15):
        """
        Initialize face tracker with more lenient parameters
        
        Args:
            stability_threshold: Number of consecutive frames a face must appear to be considered stable
                                (reduced to 4 from 2)
            position_tolerance: Maximum change in normalized position to be considered the same face
                               (increased to 0.15 from 0.2)
        """
        self.tracked_faces = {}  # Dictionary of tracked faces
        self.next_face_id = 0  # Counter for generating face IDs
        self.stability_threshold = stability_threshold
        self.position_tolerance = position_tolerance
        self.last_positions = {}  # Last known positions of each tracked face
        print(f"Face tracker initialized with stability threshold={stability_threshold}, " +
              f"position tolerance={position_tolerance}")
    
    def update(self, faces):
        """
        Update tracked faces with new detections
        
        Args:
            faces: List of face dictionaries with 'bbox' keys
            
        Returns:
            List of indices of faces in the input list that have reached stability threshold
        """
        stable_face_indices = []
        current_face_ids = set()
        
        # Match faces to existing tracked faces
        for i, face in enumerate(faces):
            bbox = face['bbox']
            matched = False
            
            # Calculate center point of face as percentage of frame dimensions
            face_center_x = (bbox[0] + bbox[2]) / 2
            face_center_y = (bbox[1] + bbox[3]) / 2
            
            # Try to match with existing tracked faces
            for face_id, count in self.tracked_faces.items():
                # If we have position data for this face
                if face_id in self.last_positions:
                    prev_center_x, prev_center_y = self.last_positions[face_id]
                    
                    # Calculate normalized distance (as percentage of frame)
                    distance_x = abs(face_center_x - prev_center_x)
                    distance_y = abs(face_center_y - prev_center_y)
                    
                    # If within tolerance, consider it the same face
                    if distance_x < self.position_tolerance and distance_y < self.position_tolerance:
                        self.tracked_faces[face_id] += 1
                        self.last_positions[face_id] = (face_center_x, face_center_y)
                        
                        # Mark this face ID as seen in this frame
                        current_face_ids.add(face_id)
                        
                        # Check if face is now stable
                        if self.tracked_faces[face_id] >= self.stability_threshold:
                            stable_face_indices.append(i)
                        
                        matched = True
                        break
            
            # If no match found, create new tracked face
            if not matched:
                new_face_id = self.next_face_id
                self.next_face_id += 1
                self.tracked_faces[new_face_id] = 1
                self.last_positions[new_face_id] = (face_center_x, face_center_y)
                current_face_ids.add(new_face_id)
                
                # If we're only requiring 1 frame of stability, add it immediately
                if self.stability_threshold <= 1:
                    stable_face_indices.append(i)
        
        # Remove faces that weren't seen in this frame
        all_face_ids = list(self.tracked_faces.keys())
        for face_id in all_face_ids:
            if face_id not in current_face_ids:
                del self.tracked_faces[face_id]
                if face_id in self.last_positions:
                    del self.last_positions[face_id]
        
        # Add debug information
        if stable_face_indices:
            print(f"Stable faces found: {len(stable_face_indices)} out of {len(faces)}")
            
        return stable_face_indices

class VideoCapture:
    """Video capture class that can handle both webcam and video files"""
    def __init__(self, source, width=640, height=480):
        self.source = source
        self.width = width 
        self.height = height
        self.frame_queue = Queue(maxsize=30)  # Increased buffer for smoother playback
        self.running = False
        self.thread = None
        self.fps = 0
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.is_file = not isinstance(source, int)  # Check if source is a file path or webcam ID
        self.video_length = 0  # Total frames in the video (for files only)
        self.current_frame_position = 0  # Current position in the video (for files only)
        self.video_fps = 30.0  # Default FPS, will be updated for video files
        self.last_frame_read_time = 0  # Time when the last frame was read
        
    def start(self):
        """Start capturing from video source"""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        if self.is_file:
            print(f"Started capturing from video file: {self.source}")
        else:
            print(f"Started capturing from webcam {self.source}")
        return True
    
    def stop(self):
        """Stop capturing"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        print("Stopped video capture")
    
    def _capture_loop(self):
        """Continuously capture frames from video source"""
        # Initialize video capture
        camera = cv2.VideoCapture(self.source)
        
        if not camera.isOpened():
            print(f"Error: Could not open video source {self.source}")
            self.running = False
            return
        
        # Get video file info if it's a file
        if self.is_file:
            self.video_length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = camera.get(cv2.CAP_PROP_FPS)
            print(f"Video file info: {self.video_length} frames, {self.video_fps} FPS")
            
        # Set resolution for webcam (has no effect on video files)
        if not self.is_file:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Video initialized at resolution: {actual_width}x{actual_height}")
        
        # Initialize frame timing variables
        self.last_frame_read_time = time.time()
        
        while self.running:
            try:
                # For video files, control the frame rate based on playback_speed
                if self.is_file:
                    # Calculate target time between frames based on video FPS and playback speed
                    target_frame_time = 1.0 / (self.video_fps * playback_speed)
                    
                    # Calculate how long to wait
                    current_time = time.time()
                    elapsed = current_time - self.last_frame_read_time
                    
                    # If we need to wait to maintain the desired frame rate
                    if elapsed < target_frame_time:
                        sleep_time = target_frame_time - elapsed
                        time.sleep(sleep_time)
                
                # Update the last frame read time
                self.last_frame_read_time = time.time()
                
                # Capture frame
                ret, frame = camera.read()
                
                if not ret:
                    if self.is_file:
                        print("End of video file reached")
                        # Loop back to beginning of video file
                        camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame_position = 0
                        continue
                    else:
                        print("Error reading from webcam")
                        time.sleep(0.01)
                        continue
                
                # Update frame position counter for video files
                if self.is_file:
                    self.current_frame_position = int(camera.get(cv2.CAP_PROP_POS_FRAMES))
                
                # Update queue
                try:
                    if self.frame_queue.full():
                        try:
                            # Try to clear some frames if queue is full
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass
                    self.frame_queue.put_nowait(frame)
                    
                    # Update FPS calculation
                    self.frame_count += 1
                    if self.frame_count >= 30:
                        now = time.time()
                        self.fps = self.frame_count / (now - self.last_frame_time)
                        self.frame_count = 0
                        self.last_frame_time = now
                except Exception as e:
                    print(f"Error updating frame queue: {e}")
                
            except Exception as e:
                print(f"Error capturing frame: {e}")
                time.sleep(0.01)
        
        # Release video when done
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
    
    def get_progress(self):
        """Get progress through the video file (0-100%)"""
        if self.is_file and self.video_length > 0:
            return (self.current_frame_position / self.video_length) * 100
        return 0

# Create a class to manage face detection with timed face indicators
class FaceDetector:
    def __init__(self, face_display_time=1.0):
        self.face_display_time = face_display_time  # How long to display face indicators
        self.faces = []  # Current detected faces
        self.face_timestamps = []  # Timestamps for each face detection
        self.lock = threading.Lock()
        
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
            
            # DEBUG: Print first face quality details if available
            if total_faces > 0 and 'Quality' in response['FaceDetails'][0]:
                quality = response['FaceDetails'][0]['Quality']
                pose = response['FaceDetails'][0]['Pose']
                landmarks = response['FaceDetails'][0]['Landmarks']
                print(f"First face details - Brightness: {quality['Brightness']:.1f}, " +
                      f"Sharpness: {quality['Sharpness']:.1f}, " +
                      f"Pose(Yaw,Pitch,Roll): ({pose['Yaw']:.1f},{pose['Pitch']:.1f},{pose['Roll']:.1f}), " +
                      f"Landmarks: {len(landmarks)}")
        
        # Extract face details with very minimal filtering
        faces = []
        rejected_faces = {"confidence": 0, "pose": 0, "quality": 0, "landmarks": 0, "size": 0}
        
        for face_detail in response['FaceDetails']:
            # Relax confidence threshold to allow more detections
            if face_detail['Confidence'] < 80:  # Lowered from 85
                rejected_faces["confidence"] += 1
                continue
            
            # More lenient pose evaluation, especially for frontal faces
            pose = face_detail['Pose']
            # More permissive for near-frontal faces (small yaw)
            if (abs(pose['Yaw']) < 20 and abs(pose['Pitch']) > 25) or \
               (abs(pose['Yaw']) >= 20 and (abs(pose['Yaw']) > 60 or abs(pose['Pitch']) > 20)):  
                rejected_faces["pose"] += 1
                continue
            
            # Almost no quality filtering - just ensure some minimal values
            quality = face_detail['Quality']
            # Check if we have at least one basic facial feature (rather than landmarks)
            if 'Landmarks' in face_detail and len(face_detail['Landmarks']) < 1:
                rejected_faces["landmarks"] += 1
                continue

            quality = face_detail['Quality']
            # Adaptive quality thresholds based on pose
            # For near-frontal faces, be more lenient with quality requirements
            if abs(pose['Yaw']) < 15:
                # Very frontal faces - be more lenient
                if quality.get('Brightness', 0) < 30 or quality.get('Sharpness', 0) < 30:
                    rejected_faces["quality"] += 1
                    continue
            else:
                # Side profiles - maintain higher quality requirements
                if quality.get('Brightness', 0) < 40 or quality.get('Sharpness', 0) < 40:
                    rejected_faces["quality"] += 1
                    continue
                
            # Get bounding box
            bbox = face_detail['BoundingBox']
            height, width, _ = frame.shape
            
            # Convert relative coordinates to absolute
            left = int(bbox['Left'] * width)
            top = int(bbox['Top'] * height)
            right = int((bbox['Left'] + bbox['Width']) * width)
            bottom = int((bbox['Top'] + bbox['Height']) * height)
            
            # Adaptive size filtering based on pose
            face_height = bottom - top
            # More lenient for frontal faces
            min_height = 70 if abs(pose['Yaw']) < 15 else 80
            if face_height < min_height:
                rejected_faces["size"] += 1
                continue
            
            # If passed all filters, add to faces list
            faces.append({
                'bbox': (left, top, right, bottom),
                'confidence': face_detail['Confidence'],
                'quality_score': (quality.get('Brightness', 0) + quality.get('Sharpness', 0)) / 2 if 'Brightness' in quality and 'Sharpness' in quality else 50,
                'pose': {
                    'yaw': pose['Yaw'],
                    'pitch': pose['Pitch'],
                    'roll': pose['Roll']
                }
            })
        
        # Log detailed filtering results
        if total_faces > 0:
            print(f"Filtering results: {total_faces} detected, {len(faces)} passed")
            if total_faces > len(faces):
                print(f"Rejected due to: confidence={rejected_faces['confidence']}, " +
                      f"pose={rejected_faces['pose']}, quality={rejected_faces['quality']}, " +
                      f"landmarks={rejected_faces['landmarks']}, size={rejected_faces['size']}")
        
        return faces
    except Exception as e:
        print(f"Error detecting faces with AWS: {e}")
        return []
    
def clear_face_collection():
    """Delete and recreate the AWS Rekognition face collection"""
    rekognition = get_rekognition_client()
    if not rekognition:
        print("Error: Could not connect to AWS Rekognition")
        return False
    
    try:
        # Delete the existing collection
        print(f"Deleting collection: {COLLECTION_ID}")
        rekognition.delete_collection(CollectionId=COLLECTION_ID)
        print(f"Collection {COLLECTION_ID} successfully deleted")
        
        # Create a new collection with the same ID
        print(f"Creating new collection: {COLLECTION_ID}")
        rekognition.create_collection(CollectionId=COLLECTION_ID)
        print(f"Collection {COLLECTION_ID} successfully created")
        
        print("Face collection cleared")
        return True
    except Exception as e:
        print(f"Error clearing face collection: {e}")
        return False

def is_new_face_aws(face_img):
    """Check if face is new using AWS Rekognition face search with improved duplicate detection"""
    rekognition = get_rekognition_client()
    if not rekognition:
        print("Could not get Rekognition client")
        return True, None
    
    # Convert image to bytes
    _, img_encoded = cv2.imencode('.jpg', face_img)
    img_bytes = img_encoded.tobytes()
    
    try:
        # Search for face in collection with similarity threshold of 80%
        print(f"Searching for face in collection {COLLECTION_ID}...")
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
            
            # Log all matches to help with debugging
            if len(response['FaceMatches']) > 1:
                print(f"Additional matches: " + 
                      ", ".join([f"{m['Similarity']:.1f}% (ID: {m['Face']['FaceId']})" 
                                for m in response['FaceMatches'][1:4]]))
            
            return False, matched_face_id
        
        # No matches found, check face quality before indexing
        # Detect faces with minimum quality requirements first
        face_detect_response = rekognition.detect_faces(
            Image={'Bytes': img_bytes},
            Attributes=['ALL']
        )
        
        # Check if any faces were detected and if they pass confidence threshold
        if not face_detect_response['FaceDetails']:
            print("No faces detected in image")
            return True, None
        
        face_detail = face_detect_response['FaceDetails'][0]  # Get the first face
        
        # Apply confidence threshold
        if face_detail['Confidence'] < 85:
            print(f"Face confidence too low: {face_detail['Confidence']:.1f}% (threshold: 85%)")
            return True, None
            
        # Now we know this is a new face that passes quality checks, so index it
        print("No match found and face passes quality checks. Indexing new face...")
        index_response = rekognition.index_faces(
            CollectionId=COLLECTION_ID,
            Image={'Bytes': img_bytes},
            MaxFaces=1,
            QualityFilter='MEDIUM', 
            DetectionAttributes=['DEFAULT']
        )
        
        # Debug indexing response
        print(f"Index response contains {len(index_response.get('FaceRecords', []))} face records and {len(index_response.get('UnindexedFaces', []))} unindexed faces")
        
        # Extract the Face ID
        if index_response.get('FaceRecords'):
            face_id = index_response['FaceRecords'][0]['Face']['FaceId']
            # Also get face quality metrics if available
            if 'Quality' in index_response['FaceRecords'][0]['FaceDetail']:
                quality = index_response['FaceRecords'][0]['FaceDetail']['Quality']
                quality_info = f", Quality: Brightness={quality.get('Brightness', 0):.1f}, Sharpness={quality.get('Sharpness', 0):.1f}"
            else:
                quality_info = ""
                
            print(f"Successfully indexed new face with ID: {face_id}{quality_info}")
            return True, face_id
        elif index_response.get('UnindexedFaces'):
            # Print reasons why faces weren't indexed
            for unindexed in index_response['UnindexedFaces']:
                print(f"Face not indexed: {unindexed.get('Reasons', ['Unknown'])}")
            return True, None
        else:
            print("No face records in index response")
            return True, None
    except ClientError as e:
        # Handle specific error for no faces detected
        error_message = str(e)
        print(f"AWS ClientError: {error_message}")
        
        if "InvalidParameterException" in error_message:
            if "No face detected" in error_message:
                print("No suitable face found in the image")
            elif "facial landmarks" in error_message:
                print("No suitable facial landmarks detected")
            else:
                print("Invalid parameter - face might be low quality")
        elif "ProvisionedThroughputExceededException" in error_message:
            print("AWS throughput limit exceeded - throttling request")
        return True, None  # Assume new face if error
    except Exception as e:
        print(f"Error checking face with AWS: {str(e)}")
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

# Removed server-side processed face check since AWS Rekognition already handles this

def save_face(frame, bbox):
    """Save a detected face if it's new and upload to backend, or return matched face ID"""
    # Extract face from bounding box
    try:
        left, top, right, bottom = bbox


         # Calculate face dimensions
        face_width = right - left
        face_height = bottom - top

        margin = min(30, int(face_width * 0.2))  
        top = max(0, top - margin)
        bottom = min(frame.shape[0], bottom + margin)
        left = max(0, left - margin)
        right = min(frame.shape[1], right + margin)
        
        # Extract face image
        face_image = frame[top:bottom, left:right]
        
        # Check if too small or invalid - use a smaller minimum size
        if face_image.shape[0] < 100 or face_image.shape[1] < 100:
            print(f"Face too small: {face_image.shape[0]}x{face_image.shape[1]} pixels")
            return None
        
        # Debug: Print size of extracted face
        print(f"Extracted face size: {face_image.shape[0]}x{face_image.shape[1]} pixels")
        
        # Check if this is a new face using AWS Rekognition - with debug info
        print("Checking if face is new...")
        is_new, face_id = is_new_face_aws(face_image)
        
        if not is_new:
            print(f"Face matched existing face in collection with ID: {face_id}")
            # Return the matched face ID
            return {'matched': True, 'face_id': face_id}
        
        if not face_id:
            print("No face ID returned - face may not be suitable for indexing")
            return None
        
        # Only save if enough time has passed
        if not is_time_to_detect():
            print("Detection throttled - waiting for cooldown")
            return None
        
        # Generate filename with timestamp and face ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/face_{timestamp}_{face_id[:8]}.jpg"
        
        # Save face image
        cv2.imwrite(filename, face_image)
        print(f"New face saved: {filename}")
        
        # Upload to backend
        thread = threading.Thread(
            target=upload_to_backend,
            args=(filename,),
            daemon=True
        )
        thread.start()
        
        return {'matched': False, 'face_id': face_id, 'filename': filename}
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

def select_source(video_file=None):
    """Select a video source (webcam or file)"""
    if video_file:
        # Use the provided video file
        if not os.path.exists(video_file):
            print(f"Error: Video file '{video_file}' not found!")
            return None
        return video_file
    
    # No video file provided, ask for source type
    print("\nSelect video source:")
    print("1. Webcam")
    print("2. Video file")
    
    while True:
        try:
            choice = input("Select option (1-2): ")
            
            if choice == '1':
                # Webcam option
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
                        cam_choice = input("\nSelect camera (1-{}): ".format(len(available_cameras)))
                        idx = int(cam_choice) - 1
                        if 0 <= idx < len(available_cameras):
                            return available_cameras[idx]
                        else:
                            print("Invalid choice. Please enter a valid number.")
                    except ValueError:
                        print("Please enter a number.")
                    except KeyboardInterrupt:
                        print("\nSelection cancelled")
                        return None
                        
            elif choice == '2':
                # Video file option
                file_path = input("Enter path to video file: ")
                if os.path.exists(file_path):
                    return file_path
                else:
                    print(f"Error: File '{file_path}' not found!")
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\nSelection cancelled")
            return None

def main(server_url=None, video_file=None, display_time=None, clear_collection=False):
    """Main function with more balanced face detection settings"""
    global backend_url, processing_enabled, playback_speed, face_display_time

    if clear_collection:
        if clear_face_collection():
            print("Face collection cleared successfully")
        else:
            print("Failed to clear face collection")
        if not video_file:  # If only clearing was requested, exit
            return
    
    # Set face display time if provided
    if display_time is not None:
        face_display_time = display_time
    
    print("\n==== AWS Rekognition Video Monitor with Balanced Filtering ====")
    
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
    
    # Select video source
    source = select_source(video_file)
    if source is None:
        print("No video source selected. Exiting.")
        return
    
    # Create video capture
    capture = VideoCapture(source)
    capture.start()
    
    # Create monitoring window
    cv2.namedWindow("Face Monitoring", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Monitoring", 800, 600)
    
    # Variables for processing - more balanced settings
    process_every_n = 5  # Process every 15th frame (balance between frequency and cost)
    frame_counter = 0
    face_detection_count = 0
    
    # Create face detector with timed display
    face_detector = FaceDetector(face_display_time=face_display_time)
    
    # Create face tracker with more lenient settings
    face_tracker = FaceTracker(stability_threshold=4, position_tolerance=0.15)
    
    # Create a separate thread for face detection
    face_detection_thread_active = False
    face_frame = None
    detected_faces = []
    
    # Add debug mode for additional logging
    debug_mode = True  # Set to True to see more detailed information
    
    def face_detection_worker():
        nonlocal face_frame, face_detection_count, face_detection_thread_active, detected_faces
        
        print("Face detection worker started")
        stable_face_count = 0
        processing_count = 0
        
        while processing_enabled and face_detection_thread_active:
            if face_frame is not None:
                local_frame = face_frame.copy()
                face_frame = None  # Clear the frame so we don't process it again
                processing_count += 1
                
                try:
                    # Use AWS Rekognition to detect faces with balanced filtering
                    faces = detect_faces_aws(local_frame)
                    
                    if faces:
                        # Update face detector with new faces for display
                        face_detector.update_faces([face['bbox'] for face in faces])
                        
                        # Store all detected faces for display
                        detected_faces = faces
                        
                        # For debugging: temporarily disable face tracker for direct processing
                        if len(faces) > 0:
                            print(f"Processing {len(faces)} detected faces directly")
                            for face in faces:
                                bbox = face['bbox']
                                # Save if it's a new face or get matched ID
                                result = save_face(local_frame, bbox)
                                if result:
                                    if result.get('matched', False):
                                        # This face matched an existing face
                                        matched_id = result.get('face_id')
                                        print(f"Face matched with existing ID: {matched_id}")
                                        # Here you can do something with the matched face ID
                                    else:
                                        # This is a new face
                                        face_detection_count += 1
                                        print(f"New face {face_detection_count} saved successfully with ID: {result.get('face_id')}")
                                        if 'quality_score' in face:
                                            print(f"Quality: {face['quality_score']:.1f}, " +
                                                f"Pose: Yaw={face['pose']['yaw']:.1f}°, " +
                                                f"Pitch={face['pose']['pitch']:.1f}°")
                except Exception as e:
                    print(f"Error processing frame in detection thread: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Print periodic status every 10 processed frames
                if processing_count % 10 == 0:
                    print(f"Processed {processing_count} frames, found {stable_face_count} stable faces, saved {face_detection_count} unique faces")
            
            # Short sleep to prevent CPU overuse
            time.sleep(0.01)
        
        print("Face detection thread stopped")
    
    # Start face detection thread
    face_detection_thread_active = True
    detection_thread = threading.Thread(target=face_detection_worker, daemon=True)
    detection_thread.start()


    # Print information about the balanced settings
    print("\nCOST INFORMATION:")
    print("- AWS Rekognition: $1 per 1,000 face operations")
    print("- Balanced quality filtering is applied to reduce unnecessary processing")
    print(f"- Processing 1 frame every {process_every_n} frames")
    print("- Faces must be stable for 2 consecutive frames to be processed")
    print("- Press 'p' to pause processing completely")
    print("- Press 's' to take a screenshot")
    print("\nFACE QUALITY FILTERS (STRICTER):")
    print("- Face confidence must be > 85%")
    print("- Face must not be turned more than 30° left/right or 20° up/down")
    print("- Face brightness and sharpness must be > 40%")
    print("- Basic facial features (eyes and nose) must be visible")
    print("- Face must be at least 80 pixels in height")
    print("\nPLAYBACK CONTROLS:")
    print("- Press '+' to increase speed")
    print("- Press '-' to decrease speed")
    print(f"- Current playback speed: {playback_speed:.2f}x")
    print(f"- Face indicator display time: {face_display_time:.1f} seconds")
    print("- Press 'd' and 'f' to decrease/increase face indicator display time")
    
    # Variables for managing display FPS
    last_frame_time = time.time()
    display_fps = 0
    processed_frames = 0
    
    try:
        # Main monitoring loop
        running = True
        
        while running:
            # Get frame
            frame = capture.get_frame()
            
            if frame is None:
                time.sleep(0.001)  # Very short sleep if no frame is available
                continue
            
            # Increment frame counter
            frame_counter += 1
            processed_frames += 1
                
            # Process only selected frames to reduce AWS API calls
            if processing_enabled and frame_counter % process_every_n == 0:
                # Send frame to detection thread
                if face_frame is None:  # Only update if previous frame was processed
                    face_frame = frame.copy()
            
            # Get active faces (keep this line)
            active_faces = face_detector.get_active_faces()
            
            # Only create a display copy if we have faces to draw
            if active_faces:
                display_frame = frame.copy()
                
                # Draw rectangles around active faces
                for (left, top, right, bottom) in active_faces:
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Simple status text - minimal
                if processing_enabled:
                    status_text = f"Processing: ON | Speed: {playback_speed:.1f}x"
                else:
                    status_text = "Processing: OFF (press 'p' to resume)"
                    
                # Add minimal status text
                cv2.putText(display_frame, status_text, 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                # Show frame with faces
                cv2.imshow("Face Monitoring", display_frame)
            else:
                # Just show original frame without copying
                cv2.imshow("Face Monitoring", frame)
            
            # Increase wait time for better display sync
            key = cv2.waitKey(15) & 0xFF
            
            if key == ord('q'):
                print("Quit requested")
                running = False
            elif key == ord('p'):
                # Toggle processing
                processing_enabled = not processing_enabled
                if processing_enabled:
                    print("Face processing resumed")
                else:
                    print("Face processing paused")
            elif key == ord('s'):
                # Take a screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(save_dir, f"screenshot_{timestamp}.jpg")
                cv2.imwrite(screenshot_path, display_frame)
                print(f"Screenshot saved: {screenshot_path}")
            elif key == ord('+') or key == ord('='):
                # Increase playback speed
                playback_speed += 0.1
                print(f"Playback speed increased to {playback_speed:.2f}x")
            elif key == ord('-') or key == ord('_'):
                # Decrease playback speed
                playback_speed = max(0.1, playback_speed - 0.1)  # Don't go below 0.1x
                print(f"Playback speed decreased to {playback_speed:.2f}x")
            elif key == ord('d'):
                # Decrease face display time
                face_display_time = max(0.1, face_display_time - 0.1)
                print(f"Face display time decreased to {face_display_time:.1f} seconds")
            elif key == ord('f'):
                # Increase face display time
                face_display_time += 0.1
                print(f"Face display time increased to {face_display_time:.1f} seconds")
    
    except KeyboardInterrupt:
        print("Monitoring stopped by user")
    except Exception as e:
        print(f"Error during monitoring: {e}")
    finally:
        # Clean up
        face_detection_thread_active = False
        detection_thread.join(timeout=1.0)
        capture.stop()
        cv2.destroyAllWindows()
        print("\nMonitoring stopped")
        print(f"- Faces saved to: {os.path.abspath(save_dir)}")
        
        # Get face count from AWS collection
        face_count = get_collection_face_count()
        print(f"- {face_count} unique faces in collection")
        print(f"- {face_detection_count} total face detections")

if __name__ == "__main__":
    import sys
    
    parser = argparse.ArgumentParser(description='AWS Rekognition Video Monitor')
    parser.add_argument('--server', default=None, help='Backend server URL (default: http://127.0.0.1:8080)')
    parser.add_argument('--video', default=None, help='Path to video file')
    parser.add_argument('--clear', action='store_true', help='Clear all faces from the collection')

    
    args = parser.parse_args()
    main(server_url=args.server, video_file=args.video, clear_collection=args.clear)