import os
import time
import argparse
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import chromedriver_autoinstaller
from dotenv import load_dotenv

# Import the original InstaRec module
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from InstaRec import ScreenCapture, main as instarec_main

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_CHECK_INTERVAL = 60  # Check if user is live every 60 seconds
DEFAULT_TARGET_USER = "stahc_m"
INSTAGRAM_URL = "https://www.instagram.com"

class InstagramLiveMonitor:
    """Monitor an Instagram user for live streams and automatically join them."""
    
    def __init__(self, target_user, check_interval=DEFAULT_CHECK_INTERVAL, headless=False, backend_url=None):
        """Initialize the Instagram Live Monitor.
        
        Args:
            target_user: Instagram username to monitor (without @)
            check_interval: How often to check if user is live (in seconds)
            headless: Whether to run browser in headless mode
            backend_url: URL for InstaRec backend server
        """
        self.target_user = target_user
        self.check_interval = check_interval
        self.headless = headless
        self.backend_url = backend_url
        self.driver = None
        self.is_monitoring = False
        self.current_live_session = False
        self.target_url = f"{INSTAGRAM_URL}/{self.target_user}/"
        self.instarec_process = None
            
        print(f"Monitor initialized for Instagram user: @{self.target_user}")
        print(f"Will check for live streams every {self.check_interval} seconds")
        print(f"Target URL: {self.target_url}")
        
    def setup_browser(self):
        """Set up and configure the browser."""
        print("Setting up browser...")
        
        # Ensure chromedriver is installed
        chromedriver_autoinstaller.install()
        
        # Configure Chrome options
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        
        # Add additional options for stability
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Initialize the WebDriver
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.maximize_window()
        
        # Open Instagram but don't log in - the user will do this manually
        self.driver.get(INSTAGRAM_URL)
        
        print("Browser setup complete")
        print("Please log in to Instagram manually and navigate to the target profile:")
        print(f"URL: {self.target_url}")

    def wait_for_target_url(self):
        """Wait until the user has navigated to the target profile URL."""
        print(f"Waiting for user to navigate to: {self.target_url}")
        
        while self.is_monitoring:
            current_url = self.driver.current_url
            # Use startswith to handle any URL parameters or fragments
            if current_url.startswith(self.target_url):
                print(f"Successfully reached target profile: {self.target_url}")
                return True
            
            # Add small delay to prevent CPU overuse
            time.sleep(2)
            
        return False

    def check_if_user_is_live(self):
        """Check if the target user is currently live streaming."""
        print(f"Checking if @{self.target_user} is live...")
        
        try:
            # We're already on the user's profile page, no need to navigate
            
            # Refined set of Live indicators that exclude script tags and focus on visible elements
            live_indicators = [
                # Standard text indicators in visible elements only
                "//span[not(self::script) and contains(text(), 'Live') and string-length(normalize-space(text())) < 20]",
                "//span[not(self::script) and contains(text(), 'LIVE') and string-length(normalize-space(text())) < 20]",
                "//div[not(self::script) and contains(text(), 'Live') and string-length(normalize-space(text())) < 20]",
                
                # Specific Instagram Live badges and indicators
                "//div[contains(@role, 'button')]//span[text()='Live']",
                "//span[text()='Live' or text()='LIVE']",
                
                # Profile page specific Live indicators
                "//header//span[text()='Live']",
                "//div[contains(@class, 'profile-pic')]//span[contains(text(), 'Live')]",
                
                # Story with live badge
                "//div[contains(@class, 'story') and .//span[text()='Live']]",
                
                # Avatar indicator
                "//div[contains(@class, 'avatar')]//span[text()='Live']"
            ]
            
            print("Searching for live indicators...")
            for i, indicator in enumerate(live_indicators):
                try:
                    live_elements = self.driver.find_elements(By.XPATH, indicator)
                    if live_elements:
                        for elem in live_elements:
                            # Verify this is a visible element with actual content
                            if elem.is_displayed() and elem.text.strip() != "":
                                elem_text = elem.text.strip()
                                # Check if it's an exact match for "Live" or contains it as a standalone word
                                if elem_text == "Live" or elem_text == "LIVE" or " Live " in f" {elem_text} ":
                                    print(f"Found live indicator with selector {i+1}: {indicator}")
                                    print(f"Element text: '{elem_text}'")
                                    print(f"Element is displayed: {elem.is_displayed()}")
                                    print(f"User @{self.target_user} is LIVE! 🔴")
                                    return True
                except NoSuchElementException:
                    continue
                except Exception as e:
                    print(f"Error with selector {i+1}: {e}")
            
            # Check for the Instagram-specific live badge in profile pictures
            try:
                # Instagram often uses a colorful ring around profile pics for live streams
                # This looks for the profile picture with a specific class that might indicate live status
                profile_indicators = [
                    "//div[contains(@class, '_aarf') and .//span[contains(text(), 'Live')]]",
                    "//canvas[contains(@class, 'live')]",
                    "//div[contains(@class, 'live-badge')]"
                ]
                
                for indicator in profile_indicators:
                    elements = self.driver.find_elements(By.XPATH, indicator)
                    if elements:
                        for elem in elements:
                            if elem.is_displayed():
                                print(f"Found Instagram-specific live indicator")
                                print(f"User @{self.target_user} is LIVE! 🔴")
                                return True
            except Exception as e:
                print(f"Error checking profile indicators: {e}")
            
            # If all checks fail, the user is probably not live
            print(f"User @{self.target_user} is not live at the moment.")
            return False
            
        except Exception as e:
            print(f"Error checking if user is live: {e}")
            print(f"Detailed error: {type(e).__name__}: {str(e)}")
            return False
            
    def join_livestream(self):
        """Join the user's livestream when detected."""
        print(f"Joining @{self.target_user}'s livestream...")
        
        try:
            # Try different selectors to find the livestream element
            join_selectors = [
                # Original selector
                "//span[contains(text(), 'Live')]/ancestor::div[3]",
                
                # Case variations
                "//span[contains(text(), 'LIVE')]/ancestor::div[3]",
                "//span[text()='LIVE']/ancestor::div[3]",
                
                # Different ancestor levels
                "//span[contains(text(), 'Live')]/ancestor::div[2]",
                "//span[contains(text(), 'Live')]/ancestor::div[4]",
                "//span[contains(text(), 'LIVE')]/ancestor::div[2]",
                "//span[contains(text(), 'LIVE')]/ancestor::div[4]",
                
                # Direct profile interactions
                "//div[contains(@role, 'button') and .//span[contains(text(), 'Live')]]",
                "//div[contains(@role, 'button') and .//span[contains(text(), 'LIVE')]]",
                
                # Header/avatar related
                "//header//span[contains(text(), 'Live')]/parent::div",
                "//header//span[contains(text(), 'LIVE')]/parent::div",
                
                # Profile picture with Live indicator
                "//img[contains(@alt, '@{self.target_user}')]/parent::div/parent::div",
                "//a[contains(@href, '/{self.target_user}') and .//span[contains(text(), 'Live')]]",
                "//a[contains(@href, '/{self.target_user}') and .//span[contains(text(), 'LIVE')]]",
                
                # Stories or avatar with Live
                "//div[contains(@role, 'button') and contains(@class, 'story')]",
                "//div[contains(@class, 'avatar')]//span[contains(text(), 'Live')]/ancestor::div[3]",
                "//div[contains(@class, 'avatar')]//span[contains(text(), 'LIVE')]/ancestor::div[3]"
            ]
            
            # Try each selector
            joined = False
            for i, selector in enumerate(join_selectors):
                try:
                    print(f"Trying join selector {i+1}: {selector}")
                    live_video = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    print(f"Found clickable element with selector {i+1}")
                    
                    # Get information about the element before clicking
                    try:
                        print(f"Element tag: {live_video.tag_name}")
                        print(f"Element text: '{live_video.text}'")
                        print(f"Element classes: {live_video.get_attribute('class')}")
                    except:
                        print("Could not get element details")
                    
                    # Click the element
                    live_video.click()
                    print(f"Clicked on live element using selector {i+1}")
                    joined = True
                    
                    # Wait for the livestream to load
                    time.sleep(7)
                    break
                except Exception as e:
                    print(f"Failed with selector {i+1}: {e}")
                    continue
            
            # If none of the selectors worked, try a more aggressive approach: look for any clickable element with "Live" text
            if not joined:
                print("Trying general approach: looking for any clickable element with Live text")
                try:
                    # Find all elements containing Live text
                    live_elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Live') or contains(text(), 'LIVE')]")
                    
                    if live_elements:
                        print(f"Found {len(live_elements)} elements with Live text, trying to click them:")
                        
                        for i, elem in enumerate(live_elements):
                            try:
                                # Try to get the closest clickable parent
                                parent_script = """
                                    var element = arguments[0];
                                    var current = element;
                                    for (var i = 0; i < 5; i++) {
                                        if (current.getAttribute('role') === 'button' || 
                                            current.tagName === 'BUTTON' || 
                                            current.tagName === 'A') {
                                            return current;
                                        }
                                        if (current.parentElement) {
                                            current = current.parentElement;
                                        } else {
                                            break;
                                        }
                                    }
                                    return element;
                                """
                                clickable_element = self.driver.execute_script(parent_script, elem)
                                
                                print(f"Element {i+1}: {clickable_element.tag_name} - '{clickable_element.text}'")
                                clickable_element.click()
                                print(f"Clicked element {i+1}")
                                
                                # Wait and check if we navigated to a new page
                                time.sleep(5)
                                
                                # Check if we're now on a video page
                                if "video" in self.driver.current_url or "/live/" in self.driver.current_url:
                                    print("Successfully navigated to live video page!")
                                    joined = True
                                    break
                            except Exception as click_error:
                                print(f"Failed to click element {i+1}: {click_error}")
                                continue
                except Exception as general_error:
                    print(f"Error in general approach: {general_error}")
            
            # Check if we successfully joined
            if not joined:
                print("Could not join the livestream with any method")
                return False
            
            # Maximize the video if possible
            try:
                fullscreen_selectors = [
                    "//button[contains(@aria-label, 'Expand')]",
                    "//button[contains(@aria-label, 'fullscreen')]",
                    "//button[contains(@aria-label, 'Full screen')]",
                    "//svg[contains(@aria-label, 'Enter Full Screen')]",
                    "//div[contains(@role, 'button') and contains(@aria-label, 'full')]"
                ]
                
                for fs_selector in fullscreen_selectors:
                    try:
                        fullscreen_button = self.driver.find_element(By.XPATH, fs_selector)
                        fullscreen_button.click()
                        print(f"Clicked fullscreen button with selector: {fs_selector}")
                        time.sleep(2)
                        break
                    except NoSuchElementException:
                        continue
            except Exception as fs_error:
                print(f"Could not find fullscreen button, continuing with normal view: {fs_error}")
            
            print("Successfully joined livestream")
            return True
            
        except Exception as e:
            print(f"Error joining livestream: {e}")
            print(f"Detailed error: {type(e).__name__}: {str(e)}")
            return False
            
    def get_stream_region(self):
        """Determine the screen region where the livestream is displayed."""
        try:
            print("Waiting for video element to be fully loaded...")
            # Give the page more time to load fully and render the video
            time.sleep(3)
            
            # Try to find the video element with multiple strategies
            video_element = None
            
            # Strategy 1: Find by video tag
            try:
                video_elements = self.driver.find_elements(By.TAG_NAME, "video")
                if video_elements:
                    # If multiple videos, try to find the largest one
                    largest_video = None
                    largest_area = 0
                    
                    for video in video_elements:
                        if video.is_displayed():
                            size = video.size
                            area = size['width'] * size['height']
                            if area > largest_area:
                                largest_area = area
                                largest_video = video
                    
                    if largest_video:
                        video_element = largest_video
                        print(f"Found video element by tag with size: {largest_video.size}")
            except Exception as e:
                print(f"Error finding video by tag: {e}")
            
            # Strategy 2: Try to find by role or aria-label if first approach failed
            if not video_element:
                try:
                    video_container = self.driver.find_element(By.XPATH, "//div[@role='dialog' and contains(@aria-label, 'live')]")
                    # Try to find video within this container
                    video_element = video_container.find_element(By.TAG_NAME, "video")
                    print("Found video element within dialog container")
                except Exception as e:
                    print(f"Error finding video container: {e}")
            
            # Strategy 3: If we still don't have a video element, look for a large container
            if not video_element:
                try:
                    # Look for large central element that might contain the video
                    containers = self.driver.find_elements(By.XPATH, "//div[contains(@style, 'width') and contains(@style, 'height')]")
                    
                    # Filter for large elements
                    largest_container = None
                    largest_area = 0
                    
                    for container in containers:
                        if container.is_displayed():
                            size = container.size
                            if size['width'] > 300 and size['height'] > 300:
                                area = size['width'] * size['height']
                                if area > largest_area:
                                    largest_area = area
                                    largest_container = container
                    
                    if largest_container:
                        video_element = largest_container
                        print(f"Using largest container as fallback with size: {largest_container.size}")
                except Exception as e:
                    print(f"Error finding container: {e}")
            
            # If we found a video element or container, calculate the region
            if video_element:
                # Get the location and size
                location = video_element.location
                size = video_element.size
                
                # Calculate initial region
                region = {
                    "left": location['x'],
                    "top": location['y'],
                    "width": size['width'],
                    "height": size['height']
                }
                
                # Get viewport dimensions
                viewport_width = self.driver.execute_script("return window.innerWidth")
                viewport_height = self.driver.execute_script("return window.innerHeight")
                
                # Simply force a much lower position - around 20% lower than before
                # Set a fixed offset of 180px from the top (was 130px before)
                minimum_top_offset = 180
                
                # Adjust the region regardless of the detected position
                # This ensures we're always well below the navigation
                height_adjusted = region["height"]
                if region["top"] < minimum_top_offset:
                    height_adjusted = region["height"] - (minimum_top_offset - region["top"])
                
                # Force the top position to our minimum offset
                region["top"] = minimum_top_offset
                region["height"] = max(300, height_adjusted)
                
                # Increase height by 20% to lengthen the bottom of the monitoring area
                region["height"] = int(region["height"] * 1.2)
                
                # Add some margin but keep within screen bounds
                margin = 20
                region["left"] = max(0, region["left"] - margin)
                region["width"] = min(viewport_width - region["left"], region["width"] + 2 * margin)
                
                print(f"Detected livestream region: {region}")
                return region
                
            # If all detection methods fail, use page dimensions to create a reasonable region
            print("Could not detect video element, using screen dimensions")
            
            # Get viewport dimensions
            viewport_width = self.driver.execute_script("return window.innerWidth")
            viewport_height = self.driver.execute_script("return window.innerHeight")
            
            # Use central portion of the screen, starting much lower (40% down from top)
            region = {
                "left": int(viewport_width * 0.25),
                "top": int(viewport_height * 0.4),  # Start 40% down (was 30%)
                "width": int(viewport_width * 0.5),
                "height": int(viewport_height * 0.4 * 1.2)  # Increased height by 20%
            }
            
            print(f"Created fallback region based on viewport: {region}")
            return region
            
        except Exception as e:
            print(f"Error determining stream region: {e}")
            print("Using default central region")
            
            # Since we couldn't detect properly, use a central region that's lower down
            return {"top": 250, "left": 400, "width": 600, "height": 540}  # Increased from 450 to 540
            
    def is_stream_still_active(self):
        """Check if the livestream is still active."""
        try:
            # Look for specific indicators that the stream has ended
            end_indicators = [
                "//div[contains(text(), 'Live Video Ended')]",
                "//span[contains(text(), 'Live Video Ended')]",
                "//div[text()='Live Video Ended']",
                "//span[text()='Live Video Ended']",
                "//div[contains(text(), 'Thank you for watching')]",
                "//span[contains(text(), 'Thank you for watching')]",
                "//div[contains(text(), 'ended')]",
                "//span[contains(text(), 'ended')]"
            ]
            
            for indicator in end_indicators:
                try:
                    end_element = self.driver.find_element(By.XPATH, indicator)
                    if end_element.is_displayed():
                        print(f"Stream ended indicator found: '{end_element.text}'")
                        return False
                except NoSuchElementException:
                    continue
            
            # If no end indicators are found, the stream is probably still active
            # But let's verify there's still a video element
            try:
                video_element = self.driver.find_element(By.TAG_NAME, "video")
                if video_element.is_displayed():
                    # Stream is still active
                    return True
            except NoSuchElementException:
                # No video element found
                print("No video element found, stream may have ended")
                return False
            
            # Default to assuming the stream is still active if we can't tell otherwise
            return True
            
        except Exception as e:
            print(f"Error checking if stream is still active: {e}")
            # Default to true if there's an error to prevent unexpected termination
            return True
            
    def return_to_profile_page(self):
        """Close the live stream and return to the user's profile page."""
        print(f"Returning to @{self.target_user}'s profile page...")
        
        try:
            # Look for the X button at the top right of the live stream
            close_button_selectors = [
                "//button[@aria-label='Close']",
                "//div[@role='button' and @aria-label='Close']",
                "//div[@role='button']//*[local-name()='svg' and contains(@aria-label, 'Close')]",
                "//button[text()='×']",  # × character
                "//button[text()='X']",
                "//div[@role='button' and contains(@class, 'close')]",
                "//div[@role='button' and contains(@class, 'exit')]",
                # Try to find by position in the top right corner
                "//div[@role='button' and (contains(@style, 'right') or contains(@class, 'right'))]"
            ]
            
            # Try each selector
            for selector in close_button_selectors:
                try:
                    close_button = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    
                    # Found a close button, click it
                    print(f"Found close button: {close_button.tag_name}")
                    close_button.click()
                    print("Clicked close button")
                    
                    # Wait for navigation to complete
                    time.sleep(2)
                    
                    # Check if we're back on the profile page
                    if self.target_user in self.driver.current_url:
                        print(f"Successfully returned to @{self.target_user}'s profile page")
                        return True
                    else:
                        print(f"Not on profile page after clicking close. Current URL: {self.driver.current_url}")
                        # Try to navigate directly to the profile
                        self.driver.get(self.target_url)
                        time.sleep(2)
                        return True
                        
                except (TimeoutException, NoSuchElementException):
                    continue
            
            # If no close button found, navigate directly to the profile
            print("No close button found, navigating directly to profile page")
            self.driver.get(self.target_url)
            time.sleep(2)
            return True
            
        except Exception as e:
            print(f"Error returning to profile page: {e}")
            # Try direct navigation as a fallback
            try:
                self.driver.get(self.target_url)
                time.sleep(2)
                return True
            except:
                return False
                
    def start_instarec_monitoring(self, region):
        """Start InstaRec monitoring for the given region."""
        print("Starting InstaRec face recognition...")
        
        try:
            # Create a temporary Python script that will run the modified InstaRec
            temp_script_path = "run_instarec.py"
            with open(temp_script_path, "w") as f:
                f.write(f"""
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import InstaRec

# Set the region directly
region = {{"left": {region["left"]}, "top": {region["top"]}, "width": {region["width"]}, "height": {region["height"]}}}

# Override the GUI method to avoid Tkinter issues
from types import MethodType

def get_screen_region_override(self):
    print("Using automatically detected region:", region)
    return region

# Create a modified version of main that skips GUI prompting
def main_override(server_url=None):
    global backend_url, processing_enabled
    
    print("\\n==== Instagram Live Face Recognition ====")
    
    # Set backend URL if provided
    if server_url:
        InstaRec.backend_url = server_url
    
    # Skip AWS checks for now to avoid errors
    print(f"Using auto-detected region: {{region}}")
    
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
                    print(f"Error in face detection thread: {{e}}")
                if processing_count % 10 == 0:
                    print(f"Processed {{processing_count}} frames, found {{face_detection_count}} new faces, matched {{matched_faces_count}}")
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
                status_text = f"Processing: ON (1 frame every {{InstaRec.process_every_n}})"
                status_color = (0, 255, 0)
            else:
                status_text = "Processing: PAUSED (press 'p' to resume)"
                status_color = (0, 0, 255)
            
            cv2.putText(display_frame, "Instagram Live Face Recognition", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {{capture.get_fps():.1f}} | Faces: {{len(active_faces)}}", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"New faces: {{face_detection_count}} | Matched: {{matched_faces_count}}", 
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
                print(f"Face processing {{status}}")
    
    except Exception as e:
        print(f"Error during monitoring: {{e}}")
        import traceback
        traceback.print_exc()
    finally:
        face_detection_thread_active = False
        if 'detection_thread' in locals():
            detection_thread.join(timeout=1.0)
        capture.stop()
        cv2.destroyAllWindows()
        print(f"Monitoring stopped - {{face_detection_count}} new faces detected")

# Apply the override
InstaRec.ScreenRegionSelector.get_screen_region = MethodType(get_screen_region_override, InstaRec.ScreenRegionSelector())
InstaRec.main = main_override

# Run InstaRec main with the server URL
backend_url = "{self.backend_url or 'http://18.217.189.106:8080'}"
try:
    InstaRec.main(server_url=backend_url)
except Exception as e:
    print(f"Error running InstaRec: {{e}}")
    import traceback
    traceback.print_exc()
""")
            
            # Launch the script in a new process
            print("Launching InstaRec in a separate process")
            import subprocess
            import sys
            
            # Use a non-blocking approach to start the process
            instarec_process = subprocess.Popen([sys.executable, temp_script_path])
            self.instarec_process = instarec_process  # Store it on self so it can be accessed later
            
            print("InstaRec monitoring process started")
            
            # Keep monitoring while the stream is active
            check_interval = 5  # Check every 5 seconds if stream is still active
            while self.is_monitoring and self.is_stream_still_active():
                time.sleep(check_interval)
            
            # Stream has ended or monitoring was stopped
            print("Livestream monitoring ended")
            self.current_live_session = False
            
            # Terminate the InstaRec process when done
            if self.instarec_process:
                try:
                    self.instarec_process.terminate()
                    print("InstaRec process terminated")
                except:
                    pass
                self.instarec_process = None
            
            # Return to profile page if the stream has ended (and we're still monitoring)
            if self.is_monitoring:
                self.return_to_profile_page()
            
            # Clean up the temporary script
            try:
                os.remove(temp_script_path)
            except:
                pass
            
        except Exception as e:
            print(f"Error in InstaRec monitoring: {e}")
            print(f"Detailed error: {type(e).__name__}: {str(e)}")
            self.current_live_session = False
    
    def monitor_loop(self):
        """Main monitoring loop with manual login flow."""
        print(f"Starting monitoring loop for @{self.target_user}")
        self.is_monitoring = True
        
        # Use a much shorter check interval when not in a live session
        check_interval_short = 5  # Check every 5 seconds when not in a live session
        
        try:
            # First, wait for the user to navigate to the target profile
            if not self.wait_for_target_url():
                print("Monitoring stopped before reaching target URL")
                return
            
            # Main monitoring loop
            while self.is_monitoring:
                # Only check for live status if we're not already in a live session
                if not self.current_live_session:
                    # First, verify we're still on the correct URL (user might have navigated away)
                    current_url = self.driver.current_url
                    if not current_url.startswith(self.target_url):
                        print(f"No longer on target URL. Current: {current_url}")
                        print(f"Waiting for user to navigate back to: {self.target_url}")
                        
                        # Wait for user to navigate back to target URL
                        if not self.wait_for_target_url():
                            print("Monitoring stopped before returning to target URL")
                            break
                    
                    # Refresh the page to check for updated live status
                    print(f"Refreshing page to check if @{self.target_user} is live...")
                    try:
                        self.driver.refresh()
                        # Wait for the page to load after refresh
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.TAG_NAME, "body"))
                        )
                        # Additional wait to ensure all dynamic content loads
                        time.sleep(2)
                        print("Page refreshed successfully")
                    except Exception as refresh_error:
                        print(f"Error refreshing page: {refresh_error}")
                    
                    # Check if user is live
                    is_live = self.check_if_user_is_live()
                
                    if is_live:
                        # Join the livestream
                        if self.join_livestream():
                            # Mark that we're in a live session
                            self.current_live_session = True
                            
                            # Get the stream region
                            region = self.get_stream_region()
                            
                            # Start InstaRec monitoring
                            self.start_instarec_monitoring(region)
                    
                    # Use the shorter check interval when not in a live session
                    time.sleep(check_interval_short)
                else:
                    # Use the standard longer interval when in a live session
                    time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("Monitoring stopped by user")
        except Exception as e:
            print(f"Error in monitoring loop: {e}")
            print(f"Detailed error: {type(e).__name__}: {str(e)}")
        finally:
            self.is_monitoring = False
            if self.driver:
                self.driver.quit()
    
    def start(self):
        """Start the Instagram Live Monitor."""
        # Setup browser
        self.setup_browser()
        
        # Start monitoring loop
        self.monitor_loop()

def main():
    """Main function to parse arguments and start monitoring."""
    parser = argparse.ArgumentParser(description="Instagram Live Stream Monitor")
    parser.add_argument("-u", "--user", default=DEFAULT_TARGET_USER,
                        help=f"Instagram user to monitor (default: {DEFAULT_TARGET_USER})")
    parser.add_argument("-i", "--interval", type=int, default=DEFAULT_CHECK_INTERVAL,
                        help=f"Check interval in seconds (default: {DEFAULT_CHECK_INTERVAL})")
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless mode (no browser UI)")
    parser.add_argument("--backend", default=None,
                        help="URL for InstaRec backend server")
    
    args = parser.parse_args()
    
    # Create and start the monitor
    monitor = InstagramLiveMonitor(
        target_user=args.user,
        check_interval=args.interval,
        headless=args.headless,
        backend_url=args.backend
    )
    
    # Print instructions before starting
    print("\n=== Instagram Live Monitor ===")
    print("1. A Chrome browser window will open")
    print("2. Log in to Instagram manually")
    print(f"3. Navigate to https://www.instagram.com/{args.user}")
    print("4. The script will automatically check for live streams")
    print("5. When a live stream is detected, it will join and start recording")
    print("6. Press Ctrl+C to quit\n")
    
    monitor.start()

if __name__ == "__main__":
    main()