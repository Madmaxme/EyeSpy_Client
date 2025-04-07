
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import InstaRec

# Set the region directly
region = {"left": 310, "top": 180, "width": 436, "height": 655}

# Override the GUI method to avoid Tkinter issues
from types import MethodType

def get_screen_region_override(self):
    print("Using automatically detected region:", region)
    return region

# Create a modified version of main that skips GUI prompting
def main_override(server_url=None):
    global backend_url, processing_enabled
    
    print("\n==== Instagram Live Face Recognition ====")
    
    # Set backend URL if provided
    if server_url:
        InstaRec.backend_url = server_url
    
    # Skip AWS checks for now to avoid errors
    print(f"Using auto-detected region: {region}")
    
    # Create screen capture directly with our region
    capture = InstaRec.ScreenCapture(region=region)
    capture.start()
    
    # Create face detector
    face_detector = InstaRec.FaceDetector(face_display_time=InstaRec.face_display_time)
    
    # Create monitoring window and position it away from the capture region
    import cv2
    cv2.namedWindow("Instagram Face Recognition", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Instagram Face Recognition", max(region["left"] + region["width"] + 50, 100), 50)
    cv2.resizeWindow("Instagram Face Recognition", 800, 600)
    
    # Set up face processing
    InstaRec.processing_enabled = True
    frame_counter = 0
    face_detection_count = 0
    matched_faces_count = 0
    face_detection_thread_active = False
    face_frame = None
    detected_faces = []
    
    # Define and start the face detection worker thread
    import threading
    
    def face_detection_worker():
        nonlocal face_frame, face_detection_count, face_detection_thread_active, detected_faces, matched_faces_count
        print("Face detection worker started")
        processing_count = 0
        while InstaRec.processing_enabled and face_detection_thread_active:
            if face_frame is not None:
                local_frame = face_frame.copy()
                face_frame = None
                processing_count += 1
                try:
                    # Detect faces with AWS Rekognition
                    faces = InstaRec.detect_faces_aws(local_frame)
                    if faces:
                        # Update face detector with new faces for display
                        face_detector.update_faces([face['bbox'] for face in faces])
                        detected_faces = faces
                        for face in faces:
                            bbox = face['bbox']
                            result = InstaRec.process_face(local_frame, bbox)
                            if result:
                                if result.get('matched', False):
                                    matched_faces_count += 1
                                else:
                                    face_detection_count += 1
                                    # Register detection to show indicator
                                    face_detector.register_detection(bbox, not result.get('matched', False))
                except Exception as e:
                    print(f"Error in face detection thread: {e}")
                if processing_count % 10 == 0:
                    print(f"Processed {processing_count} frames, found {face_detection_count} new faces, matched {matched_faces_count}")
            import time
            time.sleep(0.01)
        print("Face detection thread stopped")
    
    face_detection_thread_active = True
    detection_thread = threading.Thread(target=face_detection_worker, daemon=True)
    detection_thread.start()
    
    import time
    
    # Main monitoring loop
    running = True
    try:
        print("Starting face recognition monitoring...")
        print("Monitoring started - press 'q' to quit")
        
        while running:
            frame = capture.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame_counter += 1
            if InstaRec.processing_enabled and frame_counter % InstaRec.process_every_n == 0:
                if face_frame is None:
                    face_frame = frame.copy()
            
            active_faces = face_detector.get_active_faces()
            display_frame = frame.copy()
            display_frame = face_detector.draw_indicators(display_frame)
            
            for (left, top, right, bottom) in active_faces:
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            if InstaRec.processing_enabled:
                status_text = f"Processing: ON (1 frame every {InstaRec.process_every_n})"
                status_color = (0, 255, 0)
            else:
                status_text = "Processing: PAUSED (press 'p' to resume)"
                status_color = (0, 0, 255)
            
            cv2.putText(display_frame, "Instagram Live Face Recognition", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {capture.get_fps():.1f} | Faces: {len(active_faces)}", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"New faces: {face_detection_count} | Matched: {matched_faces_count}", 
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, status_text, 
                      (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            cv2.imshow("Instagram Face Recognition", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested")
                running = False
            elif key == ord('p'):
                InstaRec.processing_enabled = not InstaRec.processing_enabled
                status = "RESUMED" if InstaRec.processing_enabled else "PAUSED"
                print(f"Face processing {status}")
    
    except Exception as e:
        print(f"Error during monitoring: {e}")
        import traceback
        traceback.print_exc()
    finally:
        face_detection_thread_active = False
        if 'detection_thread' in locals():
            detection_thread.join(timeout=1.0)
        capture.stop()
        cv2.destroyAllWindows()
        print(f"Monitoring stopped - {face_detection_count} new faces detected")

# Apply the override
InstaRec.ScreenRegionSelector.get_screen_region = MethodType(get_screen_region_override, InstaRec.ScreenRegionSelector())
InstaRec.main = main_override

# Run InstaRec main with the server URL
backend_url = "http://35.180.226.30:8080"
try:
    InstaRec.main(server_url=backend_url)
except Exception as e:
    print(f"Error running InstaRec: {e}")
    import traceback
    traceback.print_exc()
