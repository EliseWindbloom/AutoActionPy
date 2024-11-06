# note this script doesn't not currently work correctly!!
import keyboard
import pyautogui
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import numpy as np
from pathlib import Path
import logging
from PIL import Image, ImageGrab
import os
import sys
from pynput import mouse
import win32gui
import win32con
import win32api
from PIL import ImageDraw

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recording.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class ScreenCapture:
    """Stores information about a captured screen region"""
    image: np.ndarray
    mouse_x: int
    mouse_y: int
    button: str
    filename: str
    timestamp: float
    
@dataclass
class KeyAction:
    """Stores information about keyboard actions"""
    keys: List[str]
    timestamp: float

class DemonstrationRecorder:
    def __init__(self, 
                 output_folder: str = "recorded_demo",
                 start_key: str = "f9",
                 stop_key: str = "f9",
                 pause_key: str = "f10",
                 emergency_key: str = "esc"):
        """Initialize the demonstration recorder"""
        self.output_folder = Path(output_folder)
        self.images_folder = self.output_folder / "images"
        self.start_key = start_key
        self.stop_key = stop_key
        self.emergency_key = emergency_key
        
        # Recording state
        self.is_recording = False
        self.last_action_time = 0
        self.wait_start_time = 0
        self.current_keys = []
        self.captures: List[ScreenCapture] = []
        self.key_actions: List[KeyAction] = []
        
        # Create output folders
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.images_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize screenshot counter
        self.screenshot_counter = 0

        self.pause_key = pause_key
        self.is_paused = False
        self.mouse_listener = None
        self.keyboard_buffer = []
        self.last_key_time = 0
        self.key_buffer_timeout = 3.0  # seconds
        
        # For duplicate detection
        self.image_hashes = set()
        
    def start(self):
        """Start the recording process"""
        logging.info("Starting demonstration recorder...")
        logging.info(f"Press {self.start_key} to start/stop recording")
        logging.info(f"Press {self.pause_key} to pause/resume")
        logging.info(f"Press {self.emergency_key} for emergency stop")
        
        # Register all hotkeys
        keyboard.on_press_key(self.start_key, self._toggle_recording)
        keyboard.on_press_key(self.pause_key, self._toggle_pause)
        keyboard.on_press_key(self.emergency_key, self._emergency_stop)
        
        keyboard.wait()

    def _toggle_recording(self, _):
        """Toggle recording state"""
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()
        
    def _toggle_pause(self, _):
        """Toggle pause state"""
        if not self.is_recording:
            return
            
        self.is_paused = not self.is_paused
        logging.info(f"Recording {'paused' if self.is_paused else 'resumed'}")
            
    def _start_recording(self):
        """Start a new recording session"""
        self.is_recording = True
        self.is_paused = False
        self.last_action_time = time.time()
        self.wait_start_time = time.time()
        self.captures.clear()
        self.key_actions.clear()
        self.current_keys.clear()
        self.image_hashes.clear()
        
        # Register keyboard hooks for all keys
        keyboard.on_press(self._on_key_event)  # Add this line
        keyboard.on_press_key('ctrl', self._on_ctrl_press)
        
        # Mouse position listener
        self.mouse_listener = mouse.Listener(
            on_move=self._on_move
        )
        self.mouse_listener.start()
        
        logging.info("Recording started")
        logging.info("Press CTRL to record a left mouse button press.")
        
    def _stop_recording(self):
        """Stop recording and process results"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.is_paused = False
        
        # Stop listeners
        if self.mouse_listener:
            self.mouse_listener.stop()
        keyboard.unhook_all()
        
        # Flush any remaining key sequence
        self._flush_key_sequence()
        
        logging.info("Recording stopped, processing results...")
        
        try:
            self._process_captures()
            self._generate_action_list()
            logging.info("Recording processed and saved successfully")
        except Exception as e:
            logging.error(f"Error processing recording: {str(e)}")

    def _emergency_stop(self, _):
        """Handle emergency stop"""
        if self.is_recording:
            logging.warning("Emergency stop triggered!")
            self._stop_recording()
        sys.exit(0)

    def _on_key_event(self, event):
        """Handle keyboard events"""
        if not self.is_recording or self.is_paused:
            return
            
        # Ignore recording control keys
        if event.name in (self.start_key, self.stop_key, self.emergency_key, self.pause_key):
            return

        # List of special keys that should be wrapped in curly braces
        special_keys = {
            'ctrl', 'shift', 'alt', 'enter', 'backspace', 'delete', 'tab',
            'up', 'down', 'left', 'right', 'home', 'end', 'pageup', 'pagedown',
            'insert', 'escape', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 
            'f7', 'f8', 'f9', 'f10', 'f11', 'f12'
        }

        # Format the key based on its type
        if event.name in special_keys:
            key_text = f"{{{event.name}}}"
        elif event.name == 'space':
            key_text = " "  # Just use a space character
        else:
            key_text = event.name  # Regular character keys

        # Add to current sequence
        self.current_keys.append(key_text)
        current_time = time.time()
        
        # Flush if enough time has passed since last key
        if current_time - self.last_key_time > self.key_buffer_timeout:
            self._flush_key_sequence()
        
        self.last_key_time = current_time
        self.last_action_time = current_time

    def _flush_key_sequence(self):
        """Process and clear the current key sequence"""
        if not self.current_keys:
            return

        # Join the keys into a single text command
        text = ''.join(self.current_keys)
        
        # Create a KeyAction object with timestamp
        self.key_actions.append(KeyAction(
            keys=[text],
            timestamp=time.time()
        ))
        self.current_keys.clear()
            
    def _on_ctrl_press(self, _):
        """Handle ctrl key press"""
        if not self.is_recording or self.is_paused:
            return
            
        # Get current mouse position
        x, y = pyautogui.position()
        self._capture_click(x, y, "left")
            
    def _on_move(self, x, y):
        """Handle mouse movement"""
        if not self.is_recording or self.is_paused:
            return
            
        # Optional: Record significant mouse movements
        pass
        
    def _capture_click(self, x: int, y: int, button: str):
        """Capture screen region around click with improved handling"""
        # Store original mouse position
        original_pos = pyautogui.position()
        screen_size = pyautogui.size()
        
        try:
            # Move mouse out of the way (near corner but not at corner)
            safe_x = screen_size[0] - 10  # 10 pixels from right edge
            safe_y = screen_size[1] - 10  # 10 pixels from bottom edge
            pyautogui.moveTo(safe_x, safe_y, duration=0)
            
            # Wait for any UI changes that might occur when mouse moves away
            time.sleep(1)
            
            # Take screenshot
            screen = np.array(ImageGrab.grab())
            
            # Calculate image hash for duplicate detection
            img_hash = self._calculate_image_hash(screen)
            
            # Skip if duplicate
            if img_hash in self.image_hashes:
                logging.info("Skipping duplicate image")
                return
                
            self.image_hashes.add(img_hash)
            
            # Generate filename
            self.screenshot_counter += 1
            filename = f"click_{self.screenshot_counter}_{button}.png"
            
            # Store capture data
            self.captures.append(ScreenCapture(
                image=screen,
                mouse_x=x,
                mouse_y=y,
                button=button,
                filename=filename,
                timestamp=time.time()
            ))
            
            # Restore mouse position and click
            pyautogui.moveTo(x, y, duration=0)
            time.sleep(0.1)
            pyautogui.click(x=x, y=y, button=button)
            
        finally:
            # Ensure mouse position is restored even if error occurs
            pyautogui.moveTo(original_pos[0], original_pos[1], duration=0)
            
    def _calculate_image_hash(self, image: np.ndarray) -> str:
        """Calculate perceptual hash of image for duplicate detection"""
        img = Image.fromarray(image)
        img = img.resize((8, 8), Image.LANCZOS).convert('L')
        pixels = np.array(img).flatten()
        diff = pixels[:-1] > pixels[1:]
        return ''.join(str(int(d)) for d in diff)
        
    def _find_unique_region(self, capture: ScreenCapture) -> Tuple[int, int, int, int]:
        """Find unique region around click with improved algorithm"""
        x, y = capture.mouse_x, capture.mouse_y
        screen = capture.image
        
        min_size = 40
        max_size = 400
        step = 20
        
        best_region = None
        best_uniqueness = 0
        
        for size in range(min_size, max_size + step, step):
            # Calculate region bounds with centering
            x1 = max(0, x - size//2)
            y1 = max(0, y - size//2)
            x2 = min(screen.shape[1], x1 + size)
            y2 = min(screen.shape[0], y1 + size)
            
            region = screen[y1:y2, x1:x2]
            
            # Skip if region is too small
            if region.shape[0] < 20 or region.shape[1] < 20:
                continue
                
            # Calculate uniqueness score
            result = cv2.matchTemplate(screen, region, cv2.TM_CCOEFF_NORMED)
            matches = np.where(result >= 0.95)
            uniqueness = 1.0 / len(matches[0])
            
            if uniqueness > best_uniqueness:
                best_uniqueness = uniqueness
                best_region = (x1, y1, x2-x1, y2-y1)
                
            if uniqueness > 0.5:  # Good enough threshold
                break
                
        return best_region or (max(0, x-75), max(0, y-75), 150, 150)
        
    def _process_captures(self):
        """Process and optimize captured images with improved handling"""
        processed_captures = []
        
        for capture in self.captures:
            try:
                # Find unique region
                region = self._find_unique_region(capture)
                
                # Crop image to region
                cropped = capture.image[
                    region[1]:region[1]+region[3],
                    region[0]:region[0]+region[2]
                ]
                
                # Add visual indicator of click position
                click_pos = (
                    capture.mouse_x - region[0],
                    capture.mouse_y - region[1]
                )
                
                # Convert to PIL Image for drawing
                img_pil = Image.fromarray(cropped)
                # draw = ImageDraw.Draw(img_pil)
                
                # # Draw click indicator
                # radius = 10
                # draw.ellipse(
                #     [click_pos[0]-radius, click_pos[1]-radius,
                #      click_pos[0]+radius, click_pos[1]+radius],
                #     outline='red', width=2
                # )
                
                # Save processed image
                img_path = str(self.images_folder / capture.filename)
                img_pil.save(img_path, 'PNG', optimize=True)
                
                processed_captures.append(capture)
                
            except Exception as e:
                logging.error(f"Error processing capture {capture.filename}: {str(e)}")
                continue
                
        self.captures = processed_captures
        
    def _generate_action_list(self):
        """Generate action list with improved formatting and comments"""
        actions = [
            "# Auto-generated action list",
            f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        last_time = self.wait_start_time
        
        # Combine and sort all actions
        all_actions = (
            [(c.timestamp, 'click', c) for c in self.captures] +
            [(k.timestamp, 'keys', k) for k in self.key_actions]
        )
        all_actions.sort()
        
        for timestamp, action_type, action in all_actions:
            # Add wait if needed
            wait_time = timestamp - last_time
            if wait_time >= 1.0:
                actions.append(f"wait {wait_time:.1f}")
                
            # Add action with appropriate formatting
            if action_type == 'click':
                button_comment = ""#f" # {action.button} click"
                actions.append(f"click {action.filename}{button_comment}")
            else:
                key_str = ' '.join(action.keys)
                actions.append(f'type_text "{key_str}"')
                
            last_time = timestamp
            
        # Write action list with error handling
        try:
            with open(self.output_folder / "actions.txt", "w") as f:
                f.write("\n".join(actions))
        except Exception as e:
            logging.error(f"Error saving action list: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        recorder = DemonstrationRecorder()
        recorder.start()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)
