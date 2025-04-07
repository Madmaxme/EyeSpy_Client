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
            
            # Save a screenshot to debug the issue
            debug_dir = "debug_screenshots"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            screenshot_path = f"{debug_dir}/profile_{self.target_user}_{timestamp}.png"
            self.driver.save_screenshot(screenshot_path)
            print(f"Saved profile screenshot to {screenshot_path}")
            
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
            # Save a screenshot before trying to join
            debug_dir = "debug_screenshots"
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            screenshot_path = f"{debug_dir}/before_join_{self.target_user}_{timestamp}.png"
            self.driver.save_screenshot(screenshot_path)
            print(f"Saved pre-join screenshot to {screenshot_path}")
            
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
                    
                    # Take a screenshot after clicking
                    screenshot_path = f"{debug_dir}/after_click_{self.target_user}_{timestamp}.png"
                    self.driver.save_screenshot(screenshot_path)
                    print(f"Saved post-click screenshot to {screenshot_path}")
                    
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
                                screenshot_path = f"{debug_dir}/after_general_click_{i+1}_{timestamp}.png"
                                self.driver.save_screenshot(screenshot_path)
                                
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
            # Find the video element
            video_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "video"))
            )
            
            # Get the location and size of the video element
            location = video_element.location
            size = video_element.size
            
            # Calculate the screen region
            region = {
                "left": location['x'],
                "top": location['y'],
                "width": size['width'],
                "height": size['height']
            }
            
            # Add some margin
            margin = 10
            region["left"] = max(0, region["left"] - margin)
            region["top"] = max(0, region["top"] - margin)
            region["width"] += 2 * margin
            region["height"] += 2 * margin
            
            print(f"Detected livestream region: {region}")
            return region
            
        except Exception as e:
            print(f"Error determining stream region: {e}")
            print("Using default region")
            return {"top": 200, "left": 400, "width": 600, "height": 800}
            
    def is_stream_still_active(self):
        """Check if the livestream is still active."""
        try:
            # Look for elements that indicate the stream is still active
            try:
                video_element = self.driver.find_element(By.TAG_NAME, "video")
                return True
            except NoSuchElementException:
                pass
                
            # Check for "Live" text
            try:
                live_text = self.driver.find_element(By.XPATH, "//span[contains(text(), 'Live')]")
                return True
            except NoSuchElementException:
                pass
                
            # Check for end of broadcast message
            try:
                end_message = self.driver.find_element(By.XPATH, "//span[contains(text(), 'ended') or contains(text(), 'This live video has ended')]")
                print("Livestream has ended")
                return False
            except NoSuchElementException:
                pass
                
            # If we can't determine status, assume still active
            return True
            
        except Exception as e:
            print(f"Error checking if stream is still active: {e}")
            return False
            
    def start_instarec_monitoring(self, region):
        """Start InstaRec monitoring for the given region."""
        print("Starting InstaRec face recognition...")
        
        try:
            # First, check if ScreenCapture can be configured with a region
            try:
                # Configure ScreenCapture with the region (if it accepts region parameter)
                print(f"Setting up ScreenCapture with region: {region}")
                # This is a workaround - we'll set the region as a module-level variable
                # that InstaRec can access if it's designed to check for it
                import InstaRec
                if hasattr(InstaRec, 'set_capture_region'):
                    InstaRec.set_capture_region(region)
                else:
                    # Try to set attributes directly
                    if not hasattr(InstaRec, 'capture_region'):
                        setattr(InstaRec, 'capture_region', region)
                    print("Set capture_region attribute on InstaRec module")
            except Exception as config_error:
                print(f"Note: Could not configure ScreenCapture with region: {config_error}")
                print("InstaRec will use its default region detection")
            
            # Launch InstaRec in a separate thread, without passing region parameter
            print("Launching InstaRec monitoring thread")
            instarec_thread = threading.Thread(
                target=instarec_main,
                args=(self.backend_url,),
                # Removed the region parameter from kwargs
                daemon=True
            )
            instarec_thread.start()
            
            print("InstaRec monitoring thread started")
            
            # Keep monitoring while the stream is active
            while self.is_monitoring and self.is_stream_still_active():
                time.sleep(10)
                
            print("Livestream monitoring ended")
            self.current_live_session = False
            
        except Exception as e:
            print(f"Error in InstaRec monitoring: {e}")
            print(f"Detailed error: {type(e).__name__}: {str(e)}")
            self.current_live_session = False
    
    def monitor_loop(self):
        """Main monitoring loop with manual login flow."""
        print(f"Starting monitoring loop for @{self.target_user}")
        self.is_monitoring = True
        
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
                
                # Wait before checking again
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
    monitor.start()

if __name__ == "__main__":
    main()