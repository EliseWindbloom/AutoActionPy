# version 8
import pyautogui
import time
import logging
import sys
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path
import keyboard
import re
from PIL import Image, ImageGrab
import os
import subprocess
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

@dataclass
class ActionResult:
    """Stores the result of an automation action"""
    success: bool
    message: str
    location: Optional[Tuple[int, int]] = None
    screenshot: Optional[np.ndarray] = None
    additional_data: Dict[str, Any] = None

class AutomationError(Exception):
    """Custom exception for automation errors"""
    pass

class PCAutomation:
    def __init__(self, 
                 confidence_threshold: float = 0.99,
                 action_delay: float = 0.5,
                 mouse_move_duration: float = 0.5,
                 max_retries: int = 3,
                 emergency_stop_key: str = 'esc',
                 images_folder: str = 'images',
                 stop_on_failure: bool = True):
        """
        Initialize the automation framework
        
        Args:
            confidence_threshold: Minimum confidence for image matching
            action_delay: Delay between actions in seconds
            mouse_move_duration: Duration of mouse movements
            max_retries: Maximum number of retry attempts
            emergency_stop_key: Key to trigger emergency stop
            images_folder: Folder containing image assets
            stop_on_failure: If True, stops execution when an action fails
        """
        self.confidence_threshold = confidence_threshold
        self.action_delay = action_delay
        self.mouse_move_duration = mouse_move_duration
        self.max_retries = max_retries
        self.emergency_stop_key = emergency_stop_key
        self.running = False
        self.last_matched_region = None
        self.images_folder = images_folder
        self.stop_on_failure = stop_on_failure
        
        # Create images folder if it doesn't exist
        Path(images_folder).mkdir(parents=True, exist_ok=True)
        
        # Set up PyAutoGUI
        pyautogui.PAUSE = action_delay
        pyautogui.FAILSAFE = True
        
        # Register emergency stop
        keyboard.on_press_key(emergency_stop_key, lambda _: self._emergency_stop())
        
        # Initialize action registry
        self._init_action_registry()

    def _init_action_registry(self):
        """Initialize the registry of available actions"""
        self.actions = {
            'move_to': self.move_to,
            'move_between': self.move_between,
            'jump_to': self.jump_to,
            'click': lambda *args: self.click(*args, button="left"),
            'click_right': lambda *args: self.click(*args, button="right"),
            'click_middle': lambda *args: self.click(*args, button="middle"),
            'type_text': self.type_text,
            'check_state': self.check_state,
            'wait': self.wait,
            'press_key': self.press_key,
            'double_click': self.double_click,
            'right_click': self.right_click,
            'drag_to': self.drag_to,
            'drag_between': self.drag_between,
            'scroll': self.scroll,
            'screenshot': self.take_screenshot,
            'run': self.run_program,  # New action for running programs
        }

    def _emergency_stop(self):
        """Handle emergency stop"""
        self.running = False
        logging.warning("Emergency stop triggered!")
        sys.exit(1)

    def _get_image_path(self, image_name: str) -> str:
        """Convert image name to full path, handling both relative and absolute paths"""
        # If it's already a full path or contains directory separators, use as-is
        if os.path.isabs(image_name) or ('/' in image_name) or ('\\' in image_name):
            return image_name
            
        # Otherwise, assume it's in the images folder
        return os.path.join(self.images_folder, image_name)
                
    def _find_image(self,
                    template_path: str,
                    region: Optional[Tuple[int, int, int, int]] = None,
                    confidence: Optional[float] = None) -> ActionResult:
        """
        Strict image finding that handles both transparent and solid GUI elements
        while carefully avoiding false positives in complex UI environments
        """
        try:
            # Load template image
            full_path = self._get_image_path(template_path)
            template = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if template is None:
                return ActionResult(False, f"Failed to load template: {full_path}")

            # Capture screen
            screen = np.array(ImageGrab.grab(bbox=region))
            screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

            # Initialize variables
            has_alpha = len(template.shape) == 3 and template.shape[2] == 4
            template_edges = None
            mask = None
            
            if has_alpha:
                # Handle transparent images
                alpha = template[:,:,3]
                template_bgr = template[:,:,:3]
                mask = alpha > 127
                
                # Convert to grayscale while respecting alpha
                template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
                template_gray[~mask] = 0
                
                # Calculate edges for later verification
                template_edges = cv2.Canny(template_gray, 100, 200)
                template_edges[~mask] = 0
                
                # Normalize the visible parts
                template_norm = np.zeros_like(template_gray, dtype=np.float32)
                if mask.any():
                    visible_pixels = template_gray[mask]
                    if len(visible_pixels) > 0:
                        min_val = np.min(visible_pixels)
                        max_val = np.max(visible_pixels)
                        if max_val > min_val:
                            template_norm[mask] = ((template_gray[mask] - min_val) * 255.0 / (max_val - min_val))
            else:
                # Handle solid GUI elements
                if len(template.shape) == 3:
                    template_bgr = template
                    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
                else:
                    template_gray = template
                    template_bgr = cv2.cvtColor(template_gray, cv2.COLOR_GRAY2BGR)

                # Calculate edges for verification
                template_edges = cv2.Canny(template_gray, 100, 200)
                template_norm = template_gray.astype(np.float32)

            # Get edge characteristics once
            template_edge_pixels = np.count_nonzero(template_edges)

            # Convert screen to grayscale
            screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)

            # Perform initial template matching
            if has_alpha:
                # For transparent elements, use normalized cross-correlation
                result = cv2.matchTemplate(
                    screen_gray,
                    template_norm.astype(np.uint8),
                    cv2.TM_CCORR_NORMED,
                    mask=mask.astype(np.uint8)
                )
                threshold = 0.8  # Slightly more lenient for transparent elements
            else:
                # For solid elements, use normalized difference
                result = cv2.matchTemplate(
                    screen_gray,
                    template_norm.astype(np.uint8),
                    cv2.TM_SQDIFF_NORMED
                )
                threshold = 0.2  # Lower is better for TM_SQDIFF_NORMED

            # Find potential matches
            if has_alpha:
                loc = np.where(result >= threshold)
            else:
                loc = np.where(result <= threshold)

            best_match = None
            best_score = float('-inf') if has_alpha else float('inf')
            
            # Evaluate each potential match
            for pt in zip(*loc[::-1]):
                current_score = result[pt[1], pt[0]]
                
                # Skip if we already have a better match
                if has_alpha:
                    if current_score <= best_score:
                        continue
                else:
                    if current_score >= best_score:
                        continue
                    
                # Extract region of interest
                h, w = template_norm.shape
                roi = screen_gray[pt[1]:pt[1]+h, pt[0]:pt[0]+w]
                
                if roi.shape != template_norm.shape:
                    continue
                    
                # Calculate edge match score
                roi_edges = cv2.Canny(roi, 100, 200)
                if has_alpha and mask is not None:
                    roi_edges[~mask] = 0
                
                edge_pixels = np.count_nonzero(roi_edges)
                edge_ratio = abs(edge_pixels - template_edge_pixels) / (template_edge_pixels + 1e-6)
                
                # Verify structure and color
                if has_alpha:
                    if edge_ratio > 0.3:  # Allow some edge variation for transparent elements
                        continue
                else:
                    if edge_ratio > 0.2:  # Stricter for solid elements
                        continue
                    
                    # Additional color verification for solid elements
                    if len(template.shape) == 3:
                        roi_bgr = screen_bgr[pt[1]:pt[1]+h, pt[0]:pt[0]+w]
                        hist_match = min(cv2.compareHist(
                            cv2.calcHist([template_bgr], [i], None, [32], [0,256]),
                            cv2.calcHist([roi_bgr], [i], None, [32], [0,256]),
                            cv2.HISTCMP_CORREL
                        ) for i in range(3))
                        
                        if hist_match < 0.8:  # Color distribution must be similar
                            continue

                # Update best match if needed
                if has_alpha:
                    if current_score > best_score:
                        best_score = current_score
                        best_match = pt
                else:
                    if current_score < best_score:
                        best_score = current_score
                        best_match = pt

            if best_match is not None:
                # Calculate center position
                h, w = template.shape[:2]
                center = (best_match[0] + w//2, best_match[1] + h//2)

                # Adjust for region if specified
                if region:
                    center = (center[0] + region[0], center[1] + region[1])

                self.last_matched_region = (center[0] - w//2, center[1] - h//2, w, h)
                return ActionResult(True, f"Match found", center, screen)
            
            # Fallback: Exact match if no confident match found
            result = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_SQDIFF_NORMED)
            min_val, _, min_loc, _ = cv2.minMaxLoc(result)

            # Define a tight threshold for an exact match
            exact_threshold = 1e-6
            if min_val <= exact_threshold:
                h, w = template_gray.shape[:2]
                center = (min_loc[0] + w // 2, min_loc[1] + h // 2)
                if region:
                    center = (center[0] + region[0], center[1] + region[1])
                self.last_matched_region = (center[0] - w // 2, center[1] - h // 2, w, h)
                return ActionResult(True, "Exact match found", center, screen)

            return ActionResult(False, f"No confident match found for {template_path}")

        except Exception as e:
            return ActionResult(False, f"Error in template matching: {str(e)}")


    def move_to(self, *args) -> ActionResult:
        """
        Move mouse to target position. Supports:
        - Image-based: move_to("image.png")
        - Absolute coordinates: move_to("x100", "y200")
        - Relative coordinates: move_to("x+300", "y-400")
        - Single axis: move_to("y+400")
        - Conditional: move_to("x-200", "if", "image.png")
        """
        try:
            # Parse arguments
            coords = {'x': None, 'y': None}
            image_path = None
            current_x, current_y = pyautogui.position()

            for arg in args:
                # Handle coordinate arguments
                if isinstance(arg, str):
                    if arg.startswith('x') or arg.startswith('y'):
                        axis = arg[0]
                        value = arg[1:]
                        
                        if value.startswith('+') or value.startswith('-'):
                            # Relative movement
                            delta = int(value)
                            base = current_x if axis == 'x' else current_y
                            coords[axis] = base + delta
                        else:
                            # Absolute movement
                            coords[axis] = int(value)
                    else:
                        # Assume it's an image path
                        image_path = arg

            if image_path:
                # Image-based movement
                result = self._find_image(image_path)
                if result.success and result.location:
                    target_x = result.location[0]
                    target_y = result.location[1]
                    
                    # Override with any specified coordinates
                    if coords['x'] is not None:
                        target_x = coords['x']
                    if coords['y'] is not None:
                        target_y = coords['y']
                        
                    pyautogui.moveTo(target_x, target_y, duration=self.mouse_move_duration)
                    return result
                return result
            elif coords['x'] is not None or coords['y'] is not None:
                # Coordinate-based movement
                target_x = coords['x'] if coords['x'] is not None else current_x
                target_y = coords['y'] if coords['y'] is not None else current_y
                
                pyautogui.moveTo(target_x, target_y, duration=self.mouse_move_duration)
                return ActionResult(True, f"Moved to coordinates ({target_x}, {target_y})", 
                                location=(target_x, target_y))
                
            return ActionResult(False, "Invalid arguments for move_to command")
            
        except Exception as e:
            return ActionResult(False, f"Move failed: {str(e)}")
    
    def jump_to(self, image_path: str) -> ActionResult:
        """
        Instantly jump to an image's position
        
        Args:
            image_path: Path to the target image
        """
        result = self._find_image(image_path)
        if not result.success:
            return result
            
        try:
            pyautogui.moveTo(result.location[0], result.location[1], duration=0)
            return ActionResult(True, f"Jumped to {image_path}", result.location)
        except Exception as e:
            return ActionResult(False, f"Failed to jump: {str(e)}")
    
    def move_between(self, image1_path: str, image2_path: str, percentage: str = "50") -> ActionResult:
        """
        Move mouse to a point between two images based on percentage
        percentage: str between "0" and "100", where 0 is closest to image1 and 100 closest to image2
        Default is "50" for midpoint
        """
        result1 = self._find_image(image1_path)
        if not result1.success:
            return result1
            
        result2 = self._find_image(image2_path)
        if not result2.success:
            return result2
            
        try:
            # Convert percentage to float and clamp between 0 and 100
            pct = max(0.0, min(100.0, float(percentage)))
            # Convert percentage to decimal for calculation
            decimal = pct / 100.0
        except ValueError:
            return ActionResult(False, f"Invalid percentage value: {percentage}")
            
        if result1.location and result2.location:
            # Calculate weighted position based on percentage
            x = int(result1.location[0] * (1 - decimal) + result2.location[0] * decimal)
            y = int(result1.location[1] * (1 - decimal) + result2.location[1] * decimal)
            
            # Move to the calculated point
            pyautogui.moveTo(x, y, duration=self.mouse_move_duration)
            
        return ActionResult(True, f"Moved {pct}% between {image1_path} and {image2_path}")

    def click(self, image_path: str = None, speed: float = 1, if_not: str = None, button: str = "left") -> ActionResult:
        """
        Move to image and click with specified mouse button
        
        Args:
            image_path: Path to target image (optional)
            speed: Movement speed in seconds (0 = instant jump)
            if_not: Only click if this image is not found
            button: Mouse button to click ("left", "right", or "middle")
        """
        try:
            # Check if_not condition
            if if_not:
                check_result = self._find_image(if_not)
                if check_result.success:
                    return ActionResult(False, f"Condition failed - {if_not} was found")
            
            # Select click function based on button type
            click_func = {
                "left": pyautogui.click,
                "right": pyautogui.rightClick,
                "middle": pyautogui.middleClick
            }.get(button.lower(), pyautogui.click)  # defaults to left click
            
            if image_path is None:
                # Click at current position
                click_func()
                return ActionResult(True, f"{button.capitalize()} clicked at current position")
                
            result = self._find_image(image_path)
            if not result.success:
                return result
                
            if speed > 0:
                pyautogui.moveTo(result.location[0], result.location[1], duration=speed)
            else:
                pyautogui.moveTo(result.location[0], result.location[1], duration=0)
                
            click_func()
            return ActionResult(True, f"{button.capitalize()} clicked on {image_path}", result.location)
        except Exception as e:
            return ActionResult(False, f"Failed to {button} click: {str(e)}")

    def type_text(self, text: str) -> ActionResult:
        """
        Type text with support for key combinations
        Format for key combinations: Use curly braces like '{ctrl+m}' or '{ctrl+shift+a}'
        Regular text outside braces will be typed normally
        
        Examples:
            type_text "The ctrl key"  # Types the word "ctrl"
            type_text "{ctrl+m}"      # Presses ctrl+m keys
            type_text "Press {ctrl+c} to copy"  # Types "Press " then presses ctrl+c then types " to copy"
        """
        try:
            # Split text into regular text and key combinations
            parts = re.split(r'({[^}]+})', text)
            
            for part in parts:
                if part.startswith('{') and part.endswith('}'):
                    # Handle key combination
                    keys = [key.strip().lower() for key in part[1:-1].split('+')]
                    pyautogui.hotkey(*keys)
                else:
                    # Regular text typing
                    pyautogui.typewrite(part)
                    
            return ActionResult(True, f"Typed text: {text}")
        except Exception as e:
            return ActionResult(False, f"Failed to type text: {str(e)}")

    def check_state(self, image_path: str) -> ActionResult:
        """Check if an image exists on screen"""
        return self._find_image(image_path)
    
    def _wait_for_images(self, image_paths: List[str], timeout: Optional[float] = None) -> ActionResult:
        """
        Wait for any of the specified images to appear on screen
        
        Args:
            image_paths: List of paths to images to wait for
            timeout: Maximum time to wait in seconds (None for infinite)
        """
        start_time = time.time()
        while True:
            for image_path in image_paths:
                result = self._find_image(image_path)
                if result.success:
                    return result
                    
            if timeout is not None and (time.time() - start_time) > timeout:
                return ActionResult(False, f"Timeout waiting for images: {', '.join(image_paths)}")
                    
            time.sleep(0.5)  # Short sleep to prevent CPU overload

    def wait(self, *args) -> ActionResult:
        """
        Enhanced wait function that can:
        1. Wait for a specified number of seconds: wait 5
        2. Wait for an image to appear: wait image.png
        3. Wait for an image with timeout: wait image.png 10
        4. Wait for either of multiple images: wait image1.png or image2.png
        5. Wait for either of multiple images with timeout: wait image1.png or image2.png 30
        """
        if len(args) == 0:
            return ActionResult(False, "Wait requires at least one argument")
                
        # If first argument is a number, treat as simple delay
        if args[0].replace('.', '').isdigit():
            return self._wait_seconds(float(args[0]))
        
        # Parse arguments for OR condition
        images = []
        timeout = None
        
        for arg in args:
            if arg.lower() == 'or':
                continue
            elif arg.replace('.', '').isdigit():
                timeout = float(arg)
            else:
                images.append(arg)
        
        return self._wait_for_images(images, timeout)
    
    def _wait_seconds(self, seconds: float) -> ActionResult:
        """Wait for specified seconds"""
        time.sleep(seconds)
        return ActionResult(True, f"Waited for {seconds} seconds")
    
    def run_program(self, path: str, *args) -> ActionResult:
        """
        Run a program or open a file
        
        Args:
            path: Path to the program or file
            *args: Additional arguments for the program
        """
        try:
            subprocess.Popen([path] + list(args))
            return ActionResult(True, f"Successfully launched: {path}")
        except Exception as e:
            return ActionResult(False, f"Failed to launch program: {str(e)}")

    def press_key(self, key: str) -> ActionResult:
        """Press a keyboard key"""
        try:
            pyautogui.press(key)
            return ActionResult(True, f"Pressed key: {key}")
        except Exception as e:
            return ActionResult(False, f"Failed to press key: {str(e)}")

    def double_click(self, image_path: str) -> ActionResult:
        """Double click on an image"""
        result = self._find_image(image_path)
        if result.success and result.location:
            pyautogui.doubleClick(result.location[0], result.location[1])
        return result

    def right_click(self, image_path: str) -> ActionResult:
        """Right click on an image"""
        result = self._find_image(image_path)
        if result.success and result.location:
            pyautogui.rightClick(result.location[0], result.location[1])
        return result

    def drag_to(self, image_path: str) -> ActionResult:
        """
        Drag from current mouse position to the center of the specified image
        Holds left mouse button, moves to target, then releases
        """
        # Find the target image
        result = self._find_image(image_path)
        if not result.success:
            return result
            
        if result.location:
            # Hold left mouse button at current position
            pyautogui.mouseDown(button='left')
            
            # Move to target location
            pyautogui.moveTo(result.location[0], result.location[1], duration=self.mouse_move_duration)
            
            # Release left mouse button
            pyautogui.mouseUp(button='left')
            
            return ActionResult(True, f"Dragged to {image_path}")
            
        return ActionResult(False, f"Failed to drag to {image_path}")
    
    def drag_between(self, image1_path: str, image2_path: str, percentage: str = "50") -> ActionResult:
        """
        Drag from current mouse position to a point between two images based on percentage
        
        Args:
            image1_path: Path to first reference image
            image2_path: Path to second reference image
            percentage: str between "0" and "100", where 0 is closest to image1 and 100 closest to image2
            Default is "50" for midpoint
        """
        result1 = self._find_image(image1_path)
        if not result1.success:
            return result1
            
        result2 = self._find_image(image2_path)
        if not result2.success:
            return result2
            
        try:
            # Convert percentage to float and clamp between 0 and 100
            pct = max(0.0, min(100.0, float(percentage)))
            # Convert percentage to decimal for calculation
            decimal = pct / 100.0
        except ValueError:
            return ActionResult(False, f"Invalid percentage value: {percentage}")
            
        if result1.location and result2.location:
            # Calculate weighted position based on percentage
            x = int(result1.location[0] * (1 - decimal) + result2.location[0] * decimal)
            y = int(result1.location[1] * (1 - decimal) + result2.location[1] * decimal)
            
            # Hold left mouse button at current position
            pyautogui.mouseDown(button='left')
            
            # Move to calculated point
            pyautogui.moveTo(x, y, duration=self.mouse_move_duration)
            
            # Release left mouse button
            pyautogui.mouseUp(button='left')
            
            return ActionResult(True, f"Dragged {pct}% between {image1_path} and {image2_path}")
            
        return ActionResult(False, "Failed to calculate drag position")

    def scroll(self, amount: int) -> ActionResult:
        """Scroll by amount (positive = up, negative = down)"""
        try:
            pyautogui.scroll(amount)
            return ActionResult(True, f"Scrolled by {amount}")
        except Exception as e:
            return ActionResult(False, f"Failed to scroll: {str(e)}")

    def take_screenshot(self, filename: str, *args) -> ActionResult:
        """
        Take a screenshot of the last matched region or full screen
        
        Args:
            filename: Where to save the screenshot
            *args: Optional arguments:
                  'full' - Force a full screen screenshot
        """
        try:
            # Check if full screenshot is explicitly requested
            force_full = len(args) > 0 and args[0].lower() == 'full'
            
            if force_full or not self.last_matched_region:
                screenshot = pyautogui.screenshot()
                screenshot_type = "full screen"
            else:
                screenshot = pyautogui.screenshot(region=self.last_matched_region)
                screenshot_type = "region"
            
            screenshot.save(filename)
            return ActionResult(True, f"{screenshot_type} screenshot saved to {filename}")
        except Exception as e:
            return ActionResult(False, f"Failed to take screenshot: {str(e)}")

    def parse_action_list(self, action_list: str) -> List[Tuple[str, List[str], Dict]]:
        """Parse the action list string into a list of (action, args, condition) tuples"""
        actions = []
        for line in action_list.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Split into parts but preserve quoted strings
            parts = re.findall(r'[^"\s]+|"[^"]*"', line)
            parts = [p.strip('"') for p in parts]
            
            if parts:
                # Look for conditional keywords
                condition = {}
                if 'if' in parts:
                    idx = parts.index('if')
                    if idx + 1 < len(parts):
                        condition = {'type': 'if', 'image': parts[idx + 1]}
                        parts = parts[:idx]  # Remove condition from parts
                elif 'if_not' in parts:
                    idx = parts.index('if_not')
                    if idx + 1 < len(parts):
                        condition = {'type': 'if_not', 'image': parts[idx + 1]}
                        parts = parts[:idx]  # Remove condition from parts
                
                action_name = parts[0]
                args = parts[1:]
                actions.append((action_name, args, condition))
                
        return actions

    def execute_action_list(self, action_list: str) -> List[ActionResult]:
        """
        Execute a list of actions from either a text file or direct string input
        
        Args:
            action_list: Either a string of actions or path to text file containing actions
        """
        try:
            # Check if action_list is a file path
            if action_list.endswith('.txt') and os.path.exists(action_list):
                with open(action_list, 'r') as f:
                    action_text = f.read()
            else:
                action_text = action_list

            # Parse and execute actions using existing logic
            results = []
            actions = self.parse_action_list(action_text)
            
            try:
                self.running = True
                for action_index, (action_name, args, condition) in enumerate(actions, 1):
                    if not self.running:
                        break
                        
                    # Check condition before executing action
                    if condition:
                        check_result = self._find_image(condition['image'])
                        should_execute = check_result.success if condition['type'] == 'if' else not check_result.success
                        if not should_execute:
                            results.append(ActionResult(True, f"Skipped action due to condition: {condition}"))
                            continue
                        
                    if action_name not in self.actions:
                        result = ActionResult(False, f"Unknown action: {action_name}")
                        results.append(result)
                        if self.stop_on_failure:
                            logging.error(f"Stopping execution due to failure: {result.message}")
                            break
                        continue
                        
                    for attempt in range(self.max_retries):
                        try:
                            # Enhanced action logging
                            print(f"=== Executing Action - Line {action_index}/{len(actions)} ===")
                            print(f"Action Name: {action_name}")
                            print(f"Arguments: {args}")
                            print(f"Conditions: {condition if condition else 'None'}")
                            print(f"Attempt: {attempt + 1}/{self.max_retries}")

                            result = self.actions[action_name](*args)
                            results.append(result)

                            # Enhanced result logging
                            print(f"Success: {result.success}")
                            print(f"Message: {result.message}")
                            print(f"Location: {tuple(int(x) for x in result.location) if result.location else None}")
                            print("-----------------\n")

                            if result.success:
                                break
                            if attempt < self.max_retries - 1:
                                time.sleep(self.action_delay)
                            elif self.stop_on_failure:
                                logging.error(f"Stopping execution due to failure: {result.message}")
                                return results
                        except Exception as e:
                            if attempt == self.max_retries - 1:
                                result = ActionResult(False, f"Action failed: {str(e)}")
                                results.append(result)
                                if self.stop_on_failure:
                                    logging.error(f"Stopping execution due to failure: {result.message}")
                                    return results
                    
            finally:
                self.running = False
                
            return results
            
        except Exception as e:
            return [ActionResult(False, f"Failed to execute action list: {str(e)}")]

# Example usage
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run automated actions based on image recognition')
    parser.add_argument('action_file', nargs='?', default='example_action_list.txt',
                      help='Path to action list file (default: example_action_list.txt)')
    parser.add_argument('--images_path', default='images/notepad',
                      help='Path to images folder (default: images/notepad)')
    
    args = parser.parse_args()

    # Create automation instance with custom or default paths
    auto = PCAutomation(
        confidence_threshold=0.8,
        action_delay=0.5,
        mouse_move_duration=0.5,
        max_retries=3,
        images_folder=args.images_path,
        stop_on_failure=True
    )
    
    try:
        # Read actions from file
        with open(args.action_file, 'r') as f:
            action_list = f.read()

        # Execute the actions
        results = auto.execute_action_list(action_list)
        
        # Print results
        for i, result in enumerate(results):
            logging.info(f"Action {i + 1}: {'Success' if result.success else 'Failed'} - {result.message}")

    except FileNotFoundError:
        logging.error(f"File not found: {args.action_file}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error executing actions: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
