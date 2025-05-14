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
import platform
from datetime import datetime
from queue import Queue, Empty, Full
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Import screen capture libraries based on platform
try:
    import pyautogui  # For screenshot capabilities
    
    # For window handling
    system = platform.system()
    if system == "Darwin":  # macOS
        # Mac doesn't work with pygetwindow in the same way
        MAC_MODE = True
        try:
            import Quartz
            QUARTZ_AVAILABLE = True
        except ImportError:
            QUARTZ_AVAILABLE = False
            print("Warning: For better window detection on Mac, install PyObjC:")
            print("pip install pyobjc-core pyobjc-framework-Quartz")
    else:  # Windows/Linux
        MAC_MODE = False
        import pygetwindow as gw
except ImportError:
    print("Error: Required packages not installed. Please install with:")
    print("pip install pyautogui")
    if platform.system() == "Darwin":
        print("For Mac, also consider: pip install pyobjc-core pyobjc-framework-Quartz")
    else:
        print("pip install pygetwindow")
    exit(1)

load_dotenv()

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create directory to save face screenshots

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY") 
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")  
AWS_REGION = "eu-west-1"

# AWS Collection to store face data - You can change this name if desired
COLLECTION_ID = "eyespy-faces"  

# Backend server configuration
DEFAULT_BACKEND_URL = "http://18.217.189.106:8080"
backend_url = os.environ.get("EYESPY_BACKEND_URL", DEFAULT_BACKEND_URL)

# Initialize variables
last_detection_time = 0
detection_throttle = 3.0  # Seconds between detections
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
    def __init__(self, stability_threshold=3, position_tolerance=0.15):
        """
        Initialize face tracker with more lenient parameters for Zoom
        
        Args:
            stability_threshold: Number of consecutive frames a face must appear to be considered stable
                                (reduced for Zoom participants who may be moving)
            position_tolerance: Maximum change in normalized position to be considered the same face
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

class ZoomCapture:
    """Capture class for Zoom windows with cross-platform support"""
    def __init__(self):
        self.frame_queue = Queue(maxsize=10)
        self.running = False
        self.thread = None
        self.fps = 0
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.zoom_window = None
        self.region = None  # Region to capture (left, top, width, height)
        self.manual_region = False  # Whether region was manually set
        
    def find_zoom_window(self):
        """Find the Zoom window with platform-specific methods"""
        try:
            if MAC_MODE:
                # macOS approach
                if QUARTZ_AVAILABLE:
                    # Try to find Zoom window using Quartz
                    windows = Quartz.CGWindowListCopyWindowInfo(
                        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
                        Quartz.kCGNullWindowID
                    )
                    
                    for window in windows:
                        name = window.get('kCGWindowOwnerName', '')
                        if 'zoom' in name.lower():
                            # Found Zoom window
                            self.zoom_window = {
                                'title': name,
                                'left': window.get('kCGWindowBounds', {}).get('X', 0),
                                'top': window.get('kCGWindowBounds', {}).get('Y', 0),
                                'width': window.get('kCGWindowBounds', {}).get('Width', 800),
                                'height': window.get('kCGWindowBounds', {}).get('Height', 600)
                            }
                            print(f"Found Zoom window: {self.zoom_window['title']}, " +
                                  f"Size: {self.zoom_window['width']}x{self.zoom_window['height']}")
                            self.region = (
                                self.zoom_window['left'], 
                                self.zoom_window['top'], 
                                self.zoom_window['width'], 
                                self.zoom_window['height']
                            )
                            return True
                
                # If Quartz failed or Zoom window not found, ask for manual region
                if not self.manual_region:
                    print("Zoom window not automatically detected on macOS.")
                    return self.ask_manual_region()
                return True
            else:
                # Windows/Linux approach
                zoom_titles = ["Zoom Meeting", "Zoom", "Zoom Call", "Zoom Webinar"]
                
                for title in zoom_titles:
                    zoom_windows = [w for w in gw.getAllWindows() if 
                                    title in w.title and w.visible]
                    if zoom_windows:
                        self.zoom_window = zoom_windows[0]
                        print(f"Found Zoom window: {self.zoom_window.title}, " +
                              f"Size: {self.zoom_window.width}x{self.zoom_window.height}")
                        self.region = (
                            self.zoom_window.left, 
                            self.zoom_window.top, 
                            self.zoom_window.width, 
                            self.zoom_window.height
                        )
                        return True
                
                # If Zoom window not found, ask for manual region
                print("No Zoom window found. Make sure Zoom is running with an active call.")
                return self.ask_manual_region()
        except Exception as e:
            print(f"Error finding Zoom window: {e}")
            return self.ask_manual_region()
    
    def ask_manual_region(self):
        """Ask user to manually specify the region"""
        try:
            print("\nLet's capture the Zoom meeting window manually.")
            print("Please position your Zoom window where you want it.")
            input("Press Enter when ready, then you'll have 5 seconds to position your mouse at the TOP-LEFT corner of the Zoom window...")
            
            # Give user time to position
            for i in range(5, 0, -1):
                print(f"{i}...")
                time.sleep(1)
            
            # Get top-left position
            top_left = pyautogui.position()
            print(f"Top-left recorded at {top_left.x}, {top_left.y}")
            
            input("Now position your mouse at the BOTTOM-RIGHT corner of the Zoom window and press Enter...")
            
            # Give user time to position
            for i in range(5, 0, -1):
                print(f"{i}...")
                time.sleep(1)
            
            # Get bottom-right position
            bottom_right = pyautogui.position()
            print(f"Bottom-right recorded at {bottom_right.x}, {bottom_right.y}")
            
            # Calculate region
            width = bottom_right.x - top_left.x
            height = bottom_right.y - top_left.y
            
            if width <= 0 or height <= 0:
                print("Error: Invalid selection (width or height is zero or negative)")
                return False
            
            self.region = (top_left.x, top_left.y, width, height)
            print(f"Manual capture region set: {self.region}")
            self.manual_region = True
            return True
        except Exception as e:
            print(f"Error setting manual region: {e}")
            return False
        
    def start(self):
        """Start capturing from Zoom window"""
        if self.find_zoom_window():
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            print("Started capturing Zoom window")
            return True
        return False
    
    def stop(self):
        """Stop capturing"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        print("Stopped Zoom capture")
    
    def _capture_loop(self):
        """Continuously capture Zoom window"""
        while self.running:
            try:
                # Check if we need to refresh the window position (not for manual mode)
                if not self.manual_region:
                    if (MAC_MODE and not self.region) or (not MAC_MODE and (not self.zoom_window or not self.zoom_window.visible)):
                        # Try to find the window again
                        if not self.find_zoom_window():
                            time.sleep(1)  # Wait a second before retrying
                            continue
                
                # Take the screenshot
                try:
                    screenshot = pyautogui.screenshot(region=self.region)
                except Exception as e:
                    print(f"Error capturing screen region {self.region}: {e}")
                    time.sleep(0.5)
                    continue
                
                # Convert PIL image to OpenCV format
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
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
                
                # Add a small delay to reduce CPU usage
                time.sleep(0.05)
                
            except Exception as e:
                print(f"Error capturing Zoom window: {e}")
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

def detect_faces_aws_for_zoom(frame):
    """Detect faces using AWS Rekognition with settings optimized for Zoom"""
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
            print(f"Raw detection: {total_faces} faces found in Zoom")
        
        # Extract face details with settings optimized for Zoom
        # Zoom calls often have smaller faces in gallery view
        faces = []
        rejected_faces = {"confidence": 0, "pose": 0, "quality": 0, "landmarks": 0, "size": 0}
        
        for face_detail in response['FaceDetails']:
            # Use lower confidence threshold for Zoom (faces may be smaller)
            if face_detail['Confidence'] < 75:  # Lowered from 80 for Zoom
                rejected_faces["confidence"] += 1
                continue
            
            # More permissive pose evaluation for Zoom (people may be at angles)
            pose = face_detail['Pose']
            if (abs(pose['Yaw']) > 70 or abs(pose['Pitch']) > 30):  
                rejected_faces["pose"] += 1
                continue
            
            # Lower quality requirements for Zoom
            quality = face_detail['Quality']
            if 'Landmarks' in face_detail and len(face_detail['Landmarks']) < 1:
                rejected_faces["landmarks"] += 1
                continue

            # Lower brightness/sharpness requirements for Zoom
            if quality.get('Brightness', 0) < 20 or quality.get('Sharpness', 0) < 20:
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
            
            # Lower minimum size for Zoom (faces in gallery view are smaller)
            face_height = bottom - top
            min_height = 50  # Reduced minimum height for Zoom
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
            print(f"Filtering results for Zoom: {total_faces} detected, {len(faces)} passed")
            if total_faces > len(faces):
                print(f"Rejected due to: confidence={rejected_faces['confidence']}, " +
                      f"pose={rejected_faces['pose']}, quality={rejected_faces['quality']}, " +
                      f"landmarks={rejected_faces['landmarks']}, size={rejected_faces['size']}")
        
        return faces
    except Exception as e:
        print(f"Error detecting faces in Zoom with AWS: {e}")
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
    # Extract face from bounding box
    try:
        left, top, right, bottom = bbox
        
        # Calculate face dimensions
        face_width = right - left
        face_height = bottom - top

        # Calculate adaptive margin based on face width
        margin = min(20, int(face_width * 0.15))  # Smaller margins for Zoom faces
        top = max(0, top - margin)
        bottom = min(frame.shape[0], bottom + margin)
        left = max(0, left - margin)
        right = min(frame.shape[1], right + margin)
        
        # Extract face image
        face_image = frame[top:bottom, left:right]
        
        # Check if too small or invalid - use a smaller minimum size for Zoom
        if face_image.shape[0] < 70 or face_image.shape[1] < 70:  # Reduced from 100 for Zoom
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
        
        # Upload face image directly to backend (in-memory)
        _, img_encoded = cv2.imencode('.jpg', face_image)
        thread = threading.Thread(
            target=upload_to_backend,
            args=(img_encoded.tobytes(),),
            daemon=True
        )
        thread.start()
        return {'matched': False, 'face_id': face_id}
    except Exception as e:
        print(f"Error saving face: {e}")
        return None

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

def main(server_url=None, clear_collection=False):
    """Main function"""
    global backend_url, processing_enabled
    
    print("\n==== AWS Rekognition Zoom Monitor ====")
    
    if clear_collection:
        if clear_face_collection():
            print("Face collection cleared successfully")
        else:
            print("Failed to clear face collection")
    
    # Check if AWS credentials are configured
    if AWS_ACCESS_KEY is None or AWS_SECRET_KEY is None:
        print("\nERROR: AWS credentials not configured!")
        print("Please set environment variables or use .env file for:")
        print("- AWS_ACCESS_KEY")
        print("- AWS_SECRET_KEY")
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
    
    # Create Zoom capture
    capture = ZoomCapture()
    if not capture.start():
        print("Error: Could not start capturing Zoom. Make sure Zoom is running.")
        return
    
    # Create monitoring window
    cv2.namedWindow("Zoom Monitoring", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Zoom Monitoring", 800, 600)
    
    # Initialize components for processing
    face_tracker = FaceTracker(stability_threshold=3)  # More lenient for Zoom
    face_detector = FaceDetector(face_display_time=face_display_time)
    
    # Variables for processing
    process_every_n = 15  # Process every 15th frame to reduce costs
    frame_counter = 0
    current_faces = []
    face_detection_count = 0
    matched_faces_count = 0
    
    # Create a separate thread for face detection
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
                    # Use AWS Rekognition to detect faces with Zoom-optimized parameters
                    faces = detect_faces_aws_for_zoom(local_frame)
                    
                    if faces:
                        # Update face detector with new faces for display
                        face_detector.update_faces([face['bbox'] for face in faces])
                        
                        # Store all detected faces for display
                        detected_faces = faces
                        
                        # Process faces directly
                        if len(faces) > 0:
                            print(f"Processing {len(faces)} detected faces in Zoom")
                            for face in faces:
                                bbox = face['bbox']
                                # Save if it's a new face or get matched ID
                                result = save_face(local_frame, bbox)
                                if result:
                                    if result.get('matched', False):
                                        # This face matched an existing face
                                        matched_id = result.get('face_id')
                                        print(f"Face matched with existing ID: {matched_id}")
                                        matched_faces_count += 1
                                        # Register with face detector for display
                                        face_detector.register_detection(bbox, False)
                                    else:
                                        # This is a new face
                                        face_detection_count += 1
                                        print(f"New face {face_detection_count} saved with ID: {result.get('face_id')}")
                                        if 'quality_score' in face:
                                            print(f"Quality: {face['quality_score']:.1f}, " +
                                                f"Pose: Yaw={face['pose']['yaw']:.1f}°, " +
                                                f"Pitch={face['pose']['pitch']:.1f}°")
                                        # Register with face detector for display
                                        face_detector.register_detection(bbox, True)
                except Exception as e:
                    print(f"Error processing frame in detection thread: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Print periodic status every 10 processed frames
                if processing_count % 10 == 0:
                    print(f"Processed {processing_count} frames, found {face_detection_count} unique faces, matched {matched_faces_count} times")
            
            # Short sleep to prevent CPU overuse
            time.sleep(0.01)
        
        print("Face detection thread stopped")
    
    # Start face detection thread
    face_detection_thread_active = True
    detection_thread = threading.Thread(target=face_detection_worker, daemon=True)
    detection_thread.start()
    
    # Print information about the monitoring
    print("\nZOOM MONITORING INFORMATION:")
    print("- This script captures the Zoom window and detects faces within it")
    print("- AWS Rekognition: $1 per 1,000 face operations")
    print(f"- Processing 1 frame every {process_every_n} frames to reduce costs")
    print("- Press 'p' to pause processing completely")
    print("- Press 's' to take a screenshot")
    print("\nOPTIMIZATIONS FOR ZOOM:")
    print("- Reduced minimum face size detection for gallery view")
    print("- More permissive quality thresholds for video call quality")
    print("- Improved detection parameters for multiple faces")
    print("\nMAC-SPECIFIC INFO:")
    if MAC_MODE:
        if QUARTZ_AVAILABLE:
            print("- Using Quartz for window detection")
        else:
            print("- Using manual window selection (Quartz not available)")
    
    try:
        # Main monitoring loop
        running = True
        while running:
            # Get frame from Zoom
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
            
            # Create display frame
            display_frame = frame.copy()
            
            # Let the face detector draw recognition indicators
            display_frame = face_detector.draw_indicators(display_frame)
            
            # Draw rectangles around detected faces
            for face in detected_faces:
                bbox = face['bbox']
                left, top, right, bottom = bbox
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
            cv2.putText(display_frame, "AWS Rekognition Zoom Monitor", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {capture.get_fps():.1f} | Faces: {len(detected_faces)}", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, f"New faces: {face_detection_count} | Matched: {matched_faces_count}", 
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, status_text, 
                      (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_frame, "Press 'q' to quit, 'p' to pause/resume, 's' for screenshot", 
                      (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                
            # Show frame
            cv2.imshow("Zoom Monitoring", display_frame)
            
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
            elif key == ord('s'):
                # Take a screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(save_dir, f"zoom_screenshot_{timestamp}.jpg")
                cv2.imwrite(screenshot_path, display_frame)
                print(f"Screenshot saved: {screenshot_path}")
    
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
        print(f"- {face_detection_count} new face detections")
        print(f"- {matched_faces_count} matched face detections")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AWS Rekognition Zoom Monitor')
    parser.add_argument('--server', default=None, help='Backend server URL (default: http://18.217.189.106:8080)')
    parser.add_argument('--clear', action='store_true', help='Clear all faces from the collection')
    
    args = parser.parse_args()
    main(server_url=args.server, clear_collection=args.clear)