# Record Demonstration py
# Made by Elise Windbloom
# Version 11
import keyboard
import pyautogui
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import numpy as np
from pathlib import Path
import logging
from tkinter import *
from PIL import Image, ImageGrab
from PIL import ImageDraw, ImageTk
import os
import sys
from pynput import mouse
import win32gui
import win32con
import win32api


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
    is_manually_cropped: bool = False
    
@dataclass
class KeyAction:
    """Stores information about keyboard actions"""
    keys: List[str]
    timestamp: float

class DemonstrationRecorder:
    def __init__(self, 
                 output_folder: str = "recorded",
                 start_key: str = "f9",
                 stop_key: str = "f9",
                 pause_key: str = "f10",
                 emergency_key: str = "esc",
                 mouse_left_trigger: str = "ctrl",
                 mouse_left_trigger_manual: str = "shift"):
        """Initialize the demonstration recorder"""
        self.output_folder = Path(output_folder)
        self.images_folder = self.output_folder / "images"
        self.start_key = start_key
        self.stop_key = stop_key
        self.emergency_key = emergency_key
        self.mouse_left_trigger = mouse_left_trigger
        self.mouse_left_trigger_manual = mouse_left_trigger_manual
        self.running = False
        self.keyboard_hooks = []
        self.current_keyboard_hook = None

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
        self.sequence_start_time = None
        
        # For duplicate detection
        self.image_hashes = set()

        # For manual screenshot from selected region
        self.root = None
        self.canvas = None
        self.current_rectangle = None
        self.start_x = None 
        self.start_y = None
        self.selected_region = None
        
    def start(self):
        """Start the recording process"""
        logging.info("Starting demonstration recorder...")
        logging.info(f"Press {self.start_key} to start/stop recording")
        logging.info(f"Press {self.pause_key} to pause/resume")
        logging.info(f"Press {self.emergency_key} for emergency stop")
        
        try:
            # Store these hooks so we can clean them up later
            self.keyboard_hooks = [
                keyboard.on_press_key(self.start_key, self._toggle_recording),
                keyboard.on_press_key(self.pause_key, self._toggle_pause),
                keyboard.on_press_key(self.emergency_key, self._emergency_stop),
                keyboard.on_press_key(self.mouse_left_trigger_manual, self._handle_shift_press)
            ]
            
            self.running = True
            while self.running:
                time.sleep(0.1)
                
        finally:
            self._cleanup_all_resources()

    def _cleanup_all_resources(self):
        """Clean up all resources and hooks"""
        # Stop mouse listener
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None

        # Remove current recording hook
        if self.current_keyboard_hook:
            keyboard.unhook(self.current_keyboard_hook)
            self.current_keyboard_hook = None

        # Remove all stored keyboard hooks
        for hook in self.keyboard_hooks:
            try:
                keyboard.unhook(hook)
            except:
                pass
        self.keyboard_hooks.clear()

        # Clean up GUI if needed
        self._cleanup_gui()

        # Final keyboard cleanup
        keyboard.unhook_all()

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

    def _create_recording_folder(self):
        """Create a new dated folder for this recording session"""
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        session_folder = self.output_folder / timestamp
        session_folder.mkdir(parents=True, exist_ok=True)
        self.images_folder = session_folder / "images"
        self.images_folder.mkdir(exist_ok=True)
        return session_folder
            
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

        # Create new session folder
        self.current_session_folder = self._create_recording_folder()
        
        # Store the hook so we can remove it later
        self.current_keyboard_hook = keyboard.on_press(self._on_key_event)
        keyboard.on_press_key(self.mouse_left_trigger, self._on_ctrl_press)
        
        # Mouse position listener
        self.mouse_listener = mouse.Listener(
            on_move=self._on_move
        )
        self.mouse_listener.start()
        
        logging.info("Recording started")
        mouse_key = (self.mouse_left_trigger).upper()
        logging.info(f"Press {mouse_key} to record a left mouse button press, with AUTOMATIC region capture.")
        mouse_key = (self.mouse_left_trigger_manual).upper()
        logging.info(f"Press {mouse_key} to record a left mouse button press, with MANUAL region selection.")
        
    def _stop_recording(self):
        """Stop recording and process results"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.is_paused = False
        
        # Stop listeners
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None

        if self.current_keyboard_hook:
            keyboard.unhook(self.current_keyboard_hook)
            self.current_keyboard_hook = None
        
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
        else:
            logging.info("ESC pressed while not recording - exiting script")
            self.running = False
            self._cleanup_all_resources()
            sys.exit(0)

    def _on_key_event(self, event):
        """Handle keyboard events"""
        if not self.is_recording or self.is_paused:
            return
        
        current_time = time.time()

        # Ignore recording control keys
        if event.name in (self.start_key, self.stop_key, self.emergency_key, self.pause_key,
                    self.mouse_left_trigger, self.mouse_left_trigger_manual):
            return
        
        # Always ensure sequence_start_time is set
        if not self.sequence_start_time:
            self.sequence_start_time = current_time
        
        # Flush existing sequence if enough time has passed AND we have keys to flush
        if self.current_keys and current_time - self.last_key_time > self.key_buffer_timeout:
            self._flush_key_sequence()
            self.sequence_start_time = current_time  # Reset for new sequence

        # Format the key based on its type
        special_keys = {
            'ctrl', 'left ctrl', 'right ctrl', 'shift', 'right shift', 'left shift', 'alt', 
            'enter', 'backspace', 'delete', 'tab', 'up', 'down', 'left', 'right', 'home', 'end', 
            'pageup', 'pagedown', 'insert', 'escape', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 
            'f7', 'f8', 'f9', 'f10', 'f11', 'f12'
        }

        if event.name in special_keys:
            key_text = f"{{{event.name}}}"
        elif event.name == 'space':
            key_text = " "
        else:
            key_text = event.name

        self.current_keys.append(key_text)
        self.last_key_time = current_time
        self.last_action_time = current_time

    def _flush_key_sequence(self, timestamp=None):
        """Process and clear the current key sequence"""
        if not self.current_keys:
            return

        # Join the keys into a single text command
        text = ''.join(self.current_keys)
        
        # Use provided timestamp, sequence start time, or current time as fallback
        action_timestamp = timestamp or self.sequence_start_time or time.time()
        
        # Create a KeyAction object with guaranteed timestamp
        self.key_actions.append(KeyAction(
            keys=[text],
            timestamp=action_timestamp
        ))
        
        self.current_keys.clear()
        self.sequence_start_time = None  # Reset the sequence start time
            
    def _on_ctrl_press(self, _):
        """Handle left mouse triggering (ctrl by default) key press"""
        if not self.is_recording or self.is_paused:
            return
            
        # Get current mouse position
        x, y = pyautogui.position()
        self._capture_click(x, y, "left")

    def _handle_shift_press(self, _):
        """Handle manual region selection (shift)"""
        if not self.is_recording or self.is_paused or self.root:
            return

        x, y = pyautogui.position()
        screen_size = pyautogui.size()
        original_pos = (x, y)

        try:
            # Move mouse out of way
            pyautogui.moveTo(screen_size[0] - 10, screen_size[1] - 10, duration=0)
            time.sleep(0.1)
            
            # Take full screenshot
            screen = np.array(ImageGrab.grab())
            
            # Restore mouse position
            pyautogui.moveTo(original_pos[0], original_pos[1], duration=0)

            # Show region selection GUI
            self._show_region_selector(screen, original_pos)

        except Exception as e:
            logging.error(f"Error in shift handler: {str(e)}")
            self._cleanup_gui()

    def _show_region_selector(self, screen, original_pos):
        """Show region selection GUI"""
        try:
            self.root = Tk()
            self.root.attributes('-fullscreen', True)
            self.root.attributes('-alpha', 1.0)
            self.root.withdraw()

            # Create gray overlay
            screen_pil = Image.fromarray(screen)
            gray_overlay = Image.new('RGBA', screen_pil.size, (128, 128, 128, 10))
            tinted_screen = Image.alpha_composite(screen_pil.convert('RGBA'), gray_overlay)
            
            # Setup canvas
            self.screenshot = ImageTk.PhotoImage(tinted_screen)
            self.canvas = Canvas(
                self.root,
                width=self.root.winfo_screenwidth(),
                height=self.root.winfo_screenheight(),
                highlightthickness=0
            )
            self.canvas.pack()
            self.canvas.create_image(0, 0, image=self.screenshot, anchor=NW)

            # Bind events
            self.canvas.bind("<Button-1>", self._on_mouse_down)
            self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
            self.canvas.bind("<ButtonRelease-1>", lambda e: self._on_mouse_up(e, screen, original_pos))
            self.root.bind("<Escape>", self._cleanup_gui)

            # Show window
            self.root.deiconify()
            self.root.attributes('-topmost', True)
            self.root.focus_force()
            self.root.mainloop()

        except Exception as e:
            logging.error(f"Error in region selector: {str(e)}")
            self._cleanup_gui()

    def _on_mouse_down(self, event):
        """Handle mouse down in region selector"""
        self.start_x = event.x
        self.start_y = event.y
        
        if self.current_rectangle:
            self.canvas.delete(self.current_rectangle)
        self.current_rectangle = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline='blue', width=2
        )

    def _on_mouse_drag(self, event):
        """Handle mouse drag in region selector"""
        if self.current_rectangle:
            self.canvas.coords(
                self.current_rectangle,
                self.start_x, self.start_y,
                event.x, event.y
            )

    def _on_mouse_up(self, event, screen, original_pos):
        """Handle mouse up in region selector"""
        try:
            if self.start_x is None or self.start_y is None:
                return

            x1 = min(self.start_x, event.x)
            y1 = min(self.start_y, event.y)
            x2 = max(self.start_x, event.x)
            y2 = max(self.start_y, event.y)

            # Crop the region
            region = screen[y1:y2, x1:x2]
            
            # Create capture with manual crop flag
            self._create_capture(region, original_pos[0], original_pos[1], "left", True)
            
            # For manual region captures, add extra delay and focus handling
            #time.sleep(0.5)  # Allow time for tkinter window to fully close
            #win32gui.SetForegroundWindow(win32gui.GetDesktopWindow())# Force window focus to background
            #time.sleep(0.1)  # Small delay after focus change

            # Simulate click at original position
            #pyautogui.click(x=original_pos[0], y=original_pos[1])

        except Exception as e:
            logging.error(f"Error in mouse up handler: {str(e)}")
        finally:
            self._cleanup_gui()
            time.sleep(0.1)  # Small delay after focus change
            # Simulate click at original position
            pyautogui.click(x=original_pos[0], y=original_pos[1])

    def _cleanup_gui(self, event=None):
        """Clean up GUI resources"""
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass
            self.root = None
            self.canvas = None
            self.current_rectangle = None
            self.start_x = None
            self.start_y = None
            
    def _on_move(self, x, y):
        """Handle mouse movement"""
        if not self.is_recording or self.is_paused:
            return
            
        # Optional: Record significant mouse movements
        pass
        
    def _capture_click(self, x: int, y: int, button: str):
        # Flush any pending keyboard input first if receives sudden mouse click
        if self.current_keys:
            self._flush_key_sequence(self.sequence_start_time)
        current_time = time.time()  # Capture timestamp immediately
        # Store original mouse position
        original_pos = pyautogui.position()
        screen_size = pyautogui.size()
        
        try:
            # Move mouse out of the way
            safe_x = screen_size[0] - 10
            safe_y = screen_size[1] - 10
            pyautogui.moveTo(safe_x, safe_y, duration=0)
            
            time.sleep(1)
            
            # Take screenshot
            screen = np.array(ImageGrab.grab())
            
            # Calculate image hash
            img_hash = self._calculate_image_hash(screen)
            
            # Generate filename
            self.screenshot_counter += 1
            filename = f"click_{self.screenshot_counter}_{button}.png"
            
            # Handle duplicate detection
            if img_hash in self.image_hashes:
                logging.info("Reusing existing image")
                for existing_capture in self.captures:
                    if self._calculate_image_hash(existing_capture.image) == img_hash:
                        filename = existing_capture.filename
                        break
            else:
                self.image_hashes.add(img_hash)
                
            # Store capture data with current timestamp
            self.captures.append(ScreenCapture(
                image=screen,
                mouse_x=x,
                mouse_y=y,
                button=button,
                filename=filename,
                timestamp=current_time  # Use current timestamp
            ))
            
            # Update last action time after everything is done
            self.last_action_time = current_time  # Move this here
            
            # Restore position and click
            pyautogui.moveTo(x, y, duration=0)
            time.sleep(0.1)
            pyautogui.click(x=x, y=y, button=button)
            
        finally:
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
        
    def _create_capture(self, image, x, y, button, is_manual=False):
        """Create a new capture entry"""
        self.screenshot_counter += 1
        filename = f"click_{self.screenshot_counter}_{button}.png"
        
        self.captures.append(ScreenCapture(
            image=image,
            mouse_x=x,
            mouse_y=y,
            button=button,
            filename=filename,
            timestamp=time.time(),
            is_manually_cropped=is_manual
        ))

    def _process_captures(self):
        """Modified to respect manual cropping flag"""
        processed_captures = []
        
        for capture in self.captures:
            try:
                if not capture.is_manually_cropped:
                    # Find unique region only for automatic captures
                    region = self._find_unique_region(capture)
                    cropped = capture.image[
                        region[1]:region[1]+region[3],
                        region[0]:region[0]+region[2]
                    ]
                else:
                    # Use the manually selected region as-is
                    cropped = capture.image

                # Save image
                img_pil = Image.fromarray(cropped)
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
        
        # Combine and sort all actions by timestamp
        all_actions = (
            [(c.timestamp, 'click', c) for c in self.captures] +
            [(k.timestamp, 'keys', k) for k in self.key_actions]
        )
        all_actions.sort(key=lambda x: x[0])  # Sort by timestamp
        
        for timestamp, action_type, action in all_actions:
            # Calculate wait time from last action
            wait_time = timestamp - last_time
            
            # Add wait if significant pause (>= 0.5 seconds)
            if wait_time >= self.key_buffer_timeout:
                actions.append(f"wait {wait_time:.1f}")
                
            # Add action with appropriate formatting
            if action_type == 'click':
                actions.append(f"click {action.filename}")
            else:
                key_str = ' '.join(action.keys)
                actions.append(f'type "{key_str}"')
                
            last_time = timestamp
        
        # Write action list
        with open(self.current_session_folder / "actions.txt", "w") as f:
            f.write("\n".join(actions))

if __name__ == "__main__":
    try:
        recorder = DemonstrationRecorder()
        recorder.start()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)
