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
DEFAULT_BACKEND_URL = "http://15.236.226.31:8080"
backend_url = os.environ.get("EYESPY_BACKEND_URL", DEFAULT_BACKEND_URL)

# Initialize variables
last_detection_time = 0
detection_throttle = 1.0  # Seconds between detections
processing_enabled = True  # Flag to enable/disable face processing

# Face indicator display time (in seconds)
face_display_time = 0.4  # Display face indicator for 0.4 seconds

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

# AWS Rekognition handles face tracking, no need to track face IDs locally

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

# Create a class to manage face detection with timed face indicators
class FaceDetector:
    def __init__(self, face_display_time=1.0):
        self.face_display_time = face_display_time  # How long to display face indicators
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

def save_face(frame, bbox):
    """Save a detected face if it's new and upload to backend, or return matched face ID"""
    # AWS Rekognition handles face tracking
    
    # Extract face from bounding box
    try:
        left, top, right, bottom = bbox

        # Calculate face dimensions
        face_width = right - left
        face_height = bottom - top

        # Calculate adaptive margin based on face width
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
    
    # Initialize FaceTracker and FaceDetector
    face_tracker = FaceTracker()
    face_detector = FaceDetector()
    
    # Variables for processing
    process_every_n = 20  # Process every 20th frame to reduce costs
    frame_counter = 0
    current_faces = []
    face_detection_count = 0
    matched_faces_count = 0
    
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
                
            # We need to update the face tracker only with detected faces, not with the frame
            # Note: current_faces is a list of bounding boxes (not tracking objects)
            
            # The first frame may not have any faces detected yet
            if not hasattr(face_tracker, 'tracked_objects'):
                face_tracker.tracked_objects = []
                
            # Process only selected frames to reduce AWS API calls (detection is expensive)
            if processing_enabled and frame_counter % process_every_n == 0:
                try:
                    # Use AWS Rekognition to detect faces
                    faces = detect_faces_aws(frame)
                    
                    # Update displayed faces
                    current_faces = [face['bbox'] for face in faces]
                    
                    # Process each detected face
                    for face in faces:
                        bbox = face['bbox']
                        # Save if it's a new face
                        result = save_face(frame, bbox)
                        if result:
                            if result.get('matched', False):
                                matched_faces_count += 1
                                print(f"Matched existing face ID: {result.get('face_id')}")
                            else:
                                face_detection_count += 1
                                # Signal the face detector to show recognition indicator
                                face_detector.register_detection(bbox, True)
                    
                    # Update the face detector
                    face_detector.update(frame, current_faces)
                except Exception as e:
                    print(f"Error processing frame: {e}")
            
            # Create display frame
            display_frame = frame.copy()
            
            # Let the face detector draw recognition indicators
            display_frame = face_detector.draw_indicators(display_frame)
            
            # Draw rectangles around faces - all with the same color since we're not using the tracker
            for (left, top, right, bottom) in current_faces:
                # Use a standard green color for all faces
                color = (0, 255, 0)  # Green
                    
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            
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
            cv2.putText(display_frame, f"New faces: {face_detection_count} | Matched: {matched_faces_count}", 
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
        print(f"- {face_detection_count} new face detections")
        print(f"- {matched_faces_count} matched face detections")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='AWS Rekognition Webcam Monitor')
    parser.add_argument('--server', default=None, help='Backend server URL (default: http://localhost:8080)')
    
    args = parser.parse_args()
    main(server_url=args.server)