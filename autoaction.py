# Auto Action py
# By Elise Windbloom
# Version 14 - added wait_file (with %LAST% variables) and type_file
# Version 13 - allowed other added search mode, fixed wait time for recorder
# Version 12.1 - Fixed custom action list argument adding
# Version 12 - Config file added and big bug fixes in recorder
# V11 - Added far faster and higher quality image detection
# v10 - First stable version
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
import yaml
import fnmatch

def get_default_paths(action_file: str = None, images_path: str = None) -> tuple:
    """
    Determine action file and images paths based on input arguments
    Returns tuple of (action_file_path, images_folder_path)
    """
    # Case 1: No arguments given - use example defaults
    if not action_file and not images_path:
        return (
            "actions/examples/notepad/example_action_list.txt",
            "actions/examples/notepad/images"
        )
    
    # Case 2: Action file given but no images path
    if action_file and not images_path:
        # Convert action_file to absolute path
        action_file = os.path.abspath(action_file)
        # Get directory of action file and add /images
        action_dir = os.path.dirname(action_file)
        images_path = os.path.join(action_dir, "images")
        return (action_file, images_path)
    
    # Case 3: Both paths given
    return (os.path.abspath(action_file), os.path.abspath(images_path))

def parse_arguments():
    parser = argparse.ArgumentParser(description='AutoActionPy - Image-based automation tool')
    parser.add_argument('action_file', nargs='?', help='Path to action list file')
    parser.add_argument('--images_path', help='Path to folder containing image assets')
    return parser.parse_args()

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")

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
    def __init__(self, config_path="config.yaml", images_path=None):
        """
        Initialize the automation framework
        
        Args:
            config_path: Path to config file
            images_path: Override images folder from config
        """
        # confidence_threshold: Minimum confidence for image matching
        # action_delay: Delay between actions in seconds
        # mouse_move_duration: Duration of mouse movements
        # max_retries: Maximum number of retry attempts
        # emergency_stop_key: Key to trigger emergency stop
        # images_folder: Folder containing image assets
        # stop_on_failure: If True, stops execution when an action fails

        # Load config
        config = load_config(config_path)
        autoaction_config = config.get('autoaction', {})

        # Override images_folder if specified
        if images_path:
            self.images_folder = Path(images_path)
        else:
            self.images_folder = Path(autoaction_config.get('images_folder', 'actions/examples/notepad/images'))


        # Get settings from config with fallbacks
        self.confidence_threshold = autoaction_config.get('confidence', 0.7)
        self.images_folder = Path(images_path if images_path else autoaction_config.get('images_folder', 'actions/examples/notepad/images'))
        self.screenshots_path = Path(autoaction_config.get('screenshots_folder',"recorded/screenshots"))  
        self.search_mode = autoaction_config.get('search_mode', 'normal') #normal, advanced or experimental

        self.action_delay = autoaction_config.get('action_delay',0.5)
        self.mouse_move_duration = autoaction_config.get('mouse_move_duration',0.5)
        self.max_retries = autoaction_config.get('max_retries',3)
        self.emergency_stop_key = autoaction_config.get('emergency_stop_key','esc')
        self.stop_on_failure = autoaction_config.get('stop_on_failure',True)

        self.running = False
        self.last_matched_region = None
        self.last_filename = None
        self.last_filepath = None
        
        # Create folder if it doesn't exist
        self.screenshots_path.mkdir(parents=True, exist_ok=True)
        
        # Set up PyAutoGUI
        pyautogui.PAUSE = self.action_delay
        pyautogui.FAILSAFE = True
        
        # Register emergency stop
        keyboard.on_press_key(self.emergency_stop_key, lambda _: self._emergency_stop())
        
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
            'click_double': lambda *args: self.click(*args, button="double"),
            'type': self.type_text,
            'type_file': self.type_file,
            'check_state': self.check_state,
            'wait': self.wait,
            'wait_file': self.wait_file,
            'press_key': self.press_key,
            'drag_to': self.drag_to,
            'drag_between': self.drag_between,
            'scroll': self.scroll,
            'screenshot': self.take_fullscreen_screenshot,
            'screenshot_region': self.take_region_screenshot,
            'run': self.run_program,  # New action for running programs
            'search_mode': self.set_search_mode, 
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
    
    def set_search_mode(self, mode: str) -> ActionResult:
        """
        Set the image search mode to either 'normal', 'advanced' or 'experimental'
        """
        mode = mode.lower()
        if mode not in ["normal", "advanced", "experimental"]:
            return ActionResult(False, f"Invalid search mode: {mode}. Use 'normal', 'advanced', 'experimental'")
            
        self.search_mode = mode
        return ActionResult(True, f"Search mode set to: {mode}")

    def _process_variables(self, text: str) -> str:
        """Replace special variables in text with their values"""
        if not isinstance(text, str):
            return text
            
        replacements = {
            "%LAST_SUCCESS%": str(self.last_result.success if hasattr(self, 'last_result') else False),
            "%LAST_X%": str(self.last_matched_region[0] if self.last_matched_region else 0),
            "%LAST_Y%": str(self.last_matched_region[1] if self.last_matched_region else 0),
            "%LAST_MESSAGE%": str(self.last_result.message if hasattr(self, 'last_result') else ""),
            "%LAST_FILENAME%": str(self.last_filename if self.last_filename else ""),
            "%LAST_FILEPATH%": str(self.last_filepath if self.last_filepath else "")
        }
        
        result = text
        for var, value in replacements.items():
            result = result.replace(var, value)
        return result

    def _find_image(self,
                    template_path: str,
                    region: Optional[Tuple[int, int, int, int]] = None,
                    confidence: Optional[float] = 0.7) -> ActionResult:
            #selects which function to use based on current search mode
            if self.search_mode == "experimental":
                return self._find_image_experimental(template_path,region,confidence)
            elif self.search_mode == "advanced":
                return self._find_image_advanced(template_path,region,confidence)
            return self._find_image_normal(template_path,region,confidence)


    def _find_image_normal(self,
                    template_path: str,
                    region: Optional[Tuple[int, int, int, int]] = None,
                    confidence: Optional[float] = 0.7) -> ActionResult:
        """
        Find image onscreen
        """
        try:
            method = cv2.TM_CCOEFF_NORMED

            # Load template image
            full_path = self._get_image_path(template_path)
            template = cv2.imread(full_path, 0)#cv2.IMREAD_UNCHANGED) # load in grayscale
            if template is None:
                return ActionResult(False, f"Failed to load template: {full_path}")
            h, w = template.shape
            
            # Capture screen
            screen = np.array(ImageGrab.grab(bbox=region))
            #screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
            screen_gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)# Convert screen to grayscale
            
            result = cv2.matchTemplate(screen_gray, template, method)

            # Get location/score from results
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                location = min_loc
                score = min_val  # Lower is better
            else:
                location = max_loc
                score = max_val  # Higher is better

            # Get x and y coordinates of the match
            x, y = location #gets left/top, not centers
            center = (x + (w//2), y + (h//2))

            if score >= confidence:
                # Adjust for region if specified
                if region:
                    center = (center[0] + region[0], center[1] + region[1])

                self.last_matched_region = (center[0] - w//2, center[1] - h//2, w, h)
                return ActionResult(True, f"Match found", center, screen)
            else:
                return ActionResult(False, f"No confident match found for {template_path}, best score found = {str(score)}, needed at least = {str(confidence)}")

        except Exception as e:
            return ActionResult(False, f"Error in template matching: {str(e)}")


    def _find_image_advanced(self,
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

    def _find_image_experimental(self,
                         template_path: str,
                         region: Optional[Tuple[int, int, int, int]] = None,
                         confidence: Optional[float] = None) -> ActionResult:
        """
        Experimental image finding that uses multiple matching methods
        """
        try:
            # Load and verify template
            full_path = self._get_image_path(template_path)
            template = cv2.imread(str(full_path), cv2.IMREAD_UNCHANGED)
            if template is None:
                return ActionResult(False, f"Failed to load template: {full_path}")
            
            # Prepare template image (standardize format)
            if template.shape[-1] == 4:  # Has alpha channel
                template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)
            elif len(template.shape) == 2:  # Grayscale
                template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
            
            # Ensure template is in uint8 format
            if template.dtype != 'uint8':
                template = (template / template.max() * 255).astype('uint8')

            # Take screenshot using PIL for consistency
            screen = np.array(ImageGrab.grab(bbox=region))
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
            if screen is None:
                return ActionResult(False, "Failed to capture screen")

            # Handle region
            if region:
                x, y, w, h = region
                screen = screen[y:y+h, x:x+w]

            # Preprocess both images with Gaussian blur
            screen_processed = cv2.GaussianBlur(screen, (3,3), 0)
            template_processed = cv2.GaussianBlur(template, (3,3), 0)

            matches = []
            template_height, template_width = template.shape[:2]

            # Try multiple matching methods in same order as original
            methods = [
                cv2.TM_CCOEFF_NORMED,
                cv2.TM_CCORR_NORMED,
                cv2.TM_SQDIFF_NORMED
            ]

            conf_threshold = confidence if confidence is not None else self.confidence_threshold

            for method in methods:
                result = cv2.matchTemplate(
                    screen_processed, 
                    template_processed, 
                    method
                )

                # Handle different comparison methods
                if method == cv2.TM_SQDIFF_NORMED:
                    match_values = 1 - result
                else:
                    match_values = result

                # Find matches above confidence threshold
                locations = np.where(match_values >= conf_threshold)
                
                for pt in zip(*locations[::-1]):
                    x, y = pt
                    conf = match_values[y, x]
                    
                    # Calculate center coordinates
                    x_center = x + template_width // 2
                    y_center = y + template_height // 2
                    
                    # Add region offset if specified
                    if region:
                        x_center += region[0]
                        y_center += region[1]
                    
                    # Check for duplicates using same threshold as original
                    is_duplicate = False
                    for existing_match in matches:
                        if (abs(x_center - existing_match[0]) <= 10 and 
                            abs(y_center - existing_match[1]) <= 10):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        matches.append((x_center, y_center, float(conf)))
                        break  # Only take first match per method

                if matches:  # If we found a match with this method, try next method
                    break

            if matches:
                # Sort by confidence and take best match
                matches.sort(key=lambda x: x[2], reverse=True)
                best_match = matches[0]
                
                # Update last matched region
                self.last_matched_region = (
                    best_match[0] - template_width//2,
                    best_match[1] - template_height//2,
                    template_width,
                    template_height
                )
                
                return ActionResult(
                    True,
                    f"Match found with confidence {best_match[2]:.3f}",
                    (best_match[0], best_match[1]),
                    screen
                )

            return ActionResult(False, f"No confident match found for {template_path}")

        except Exception as e:
            return ActionResult(False, f"Error in experimental template matching: {str(e)}")


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
            button: Mouse button to click ("left", "right", "middle" or "double")
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
                "middle": pyautogui.middleClick,
                "double": pyautogui.doubleClick
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
        Type text with support for key combinations and variables
        Format for key combinations: Use curly braces like '{ctrl+m}' or '{ctrl+shift+a}'
        Regular text outside braces will be typed normally
        
        Examples:
            type "The ctrl key"  # Types the word "ctrl"
            type "{ctrl+m}"      # Presses ctrl+m keys
            type "Press {ctrl+c} to copy"  # Types "Press " then presses ctrl+c then types " to copy"
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
        
    def type_file(self, filepath: str) -> ActionResult:
        """
        Read and type the contents of a text file
        Args:
            filepath: Path to the text file to read and type
        """
        try:
            # Read the file
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Use existing type_text method to handle the content
            # This ensures we maintain support for key combinations and variables
            return self.type_text(content)
            
        except FileNotFoundError:
            return ActionResult(False, f"File not found: {filepath}")
        except Exception as e:
            return ActionResult(False, f"Failed to type file contents: {str(e)}")

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
    
    def wait_file(self, path_pattern: str, timeout: float = 30) -> ActionResult:
        """
        Wait for file changes/creation in a directory with optional file type filtering
        
        Args:
            path_pattern: Path with optional wildcard (e.g. "C:/path/*.wav" or "C:/path")
            timeout: Maximum time to wait in seconds
        """
        try:
            # Split path and pattern
            if '*' in path_pattern:
                directory = os.path.dirname(path_pattern)
                pattern = os.path.basename(path_pattern)
            else:
                directory = path_pattern
                pattern = '*'
                
            if not os.path.exists(directory):
                return ActionResult(False, f"Directory not found: {directory}")
                
            # Get initial state
            initial_files = {}
            for f in os.listdir(directory):
                if fnmatch.fnmatch(f, pattern):
                    path = os.path.join(directory, f)
                    initial_files[f] = os.path.getmtime(path)
                    
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                time.sleep(0.1)  # Reduce CPU usage
                
                # Check for changes
                current_files = {}
                for f in os.listdir(directory):
                    if fnmatch.fnmatch(f, pattern):
                        path = os.path.join(directory, f)
                        current_files[f] = os.path.getmtime(path)
                        
                        # Check if file is new or modified
                        if (f not in initial_files or 
                            current_files[f] > initial_files[f]):
                            
                            self.last_filename = f
                            self.last_filepath = path
                            return ActionResult(
                                True,
                                f"File changed/created: {f}",
                                additional_data={
                                    'filename': f,
                                    'filepath': path
                                }
                            )
                            
            return ActionResult(False, "Timeout waiting for file changes")
            
        except Exception as e:
            return ActionResult(False, f"Error monitoring files: {str(e)}")
    
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
        

    def take_fullscreen_screenshot(self, filename: str) -> ActionResult:
        """Take a full screen screenshot"""
        try:
            screenshot = ImageGrab.grab()
            # Create full path
            filepath = self.screenshots_path / filename
            screenshot.save(filepath)
            return ActionResult(True, f"Saved full screenshot to {filepath}")
        except Exception as e:
            return ActionResult(False, f"Failed to take screenshot: {str(e)}")

    def take_region_screenshot(self, filename: str) -> ActionResult:
        """Take a screenshot of the last matched region, falls back to full screen"""
        try:
            if self.last_matched_region:
                screenshot = ImageGrab.grab(bbox=self.last_matched_region)
            else:
                screenshot = ImageGrab.grab()
                logging.warning("No region available, taking full screenshot instead")
            
            # Create full path
            filepath = self.screenshots_path / filename
            screenshot.save(filepath)
            return ActionResult(True, f"Saved {'region' if self.last_matched_region else 'full'} screenshot to {filepath}")
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

                    # Process variables in condition if it exists
                    if condition:
                        condition['image'] = self._process_variables(condition['image'])
                    # Process variables in arguments
                    # Variables: %LAST_FILENAME%, %LAST_FILEPATH%, %LAST_SUCCESS%, %LAST_X%, %LAST_Y%, %LAST_MESSAGE%
                    processed_args = [self._process_variables(arg) for arg in args]
                        
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
                            print(f"Arguments: {processed_args}")
                            print(f"Conditions: {condition if condition else 'None'}")
                            print(f"Attempt: {attempt + 1}/{self.max_retries}")

                            result = self.actions[action_name](*processed_args)
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

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Get paths based on arguments
    action_file, images_path = get_default_paths(args.action_file, args.images_path)
    
    # Create automation instance with possibly overridden images path
    auto = PCAutomation(config_path="config.yaml", images_path=images_path)
    
    try:
        # Read actions from file
        with open(action_file, 'r') as f:
            action_list = f.read()
        
        # Execute the actions
        results = auto.execute_action_list(action_list)
        
        # Print results
        for i, result in enumerate(results):
            logging.info(f"Action {i + 1}: {'Success' if result.success else 'Failed'} - {result.message}")

    except FileNotFoundError:
        logging.error(f"File not found: {action_file}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error executing actions: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
