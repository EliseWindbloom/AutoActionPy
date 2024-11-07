import cv2
import numpy as np
import pyautogui
import time
from pathlib import Path
import logging

class ImageFinder:
    def __init__(self, confidence=0.8):
        """
        Initialize ImageFinder with default confidence threshold
        """
        self.confidence = confidence
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def take_screenshot(self):
        """Capture screen and convert to CV2 format"""
        try:
            screenshot = pyautogui.screenshot()
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.logger.error(f"Screenshot failed: {e}")
            return None

    def preprocess_image(self, image):
        """
        Preprocess image to handle transparency and improve matching
        """
        # Convert to uint8 type if needed
        # if image.dtype != np.uint8:
        #     image = (image * 255).astype(np.uint8)

        # Convert to RGBA if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Basic preprocessing
        processed = cv2.GaussianBlur(image, (3,3), 0)
        return processed
    
    # Function to convert image to uint8 and standardize format for cv2
    def prepare_image(self, image):
        # Check if the image has an alpha channel and remove it if so
        if image.shape[-1] == 4:  # Assuming shape is (H, W, 4)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif len(image.shape) == 2:  # Grayscale image
            # Convert to BGR to ensure compatibility, but still uint8
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Ensure image is in uint8 format
        if image.dtype != 'uint8':
            image = (image / image.max() * 255).astype('uint8')

        return image

    def find_on_screen(self, template_path, region=None, multi_match=False):
        """
        Find template image on screen using multiple methods
        
        Args:
            template_path: Path to template image
            region: (x, y, width, height) tuple for search region
            multi_match: Whether to find multiple matches
            
        Returns:
            List of (x_center, y_center, confidence) tuples for matches
        """
        try:
            # Load and verify template
            template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
            if template is None:
                raise ValueError(f"Failed to load template: {template_path}")
            
            # converts loaded image to right format for cv2 if needed
            template = self.prepare_image(template)

            # Take screenshot
            screen = self.take_screenshot()
            if screen is None:
                return []

            # Handle region
            if region:
                x, y, w, h = region
                screen = screen[y:y+h, x:x+w]

            # Ensure both images are the same type
            # screen = screen.astype(np.uint8)
            # template = template.astype(np.uint8)

            # Preprocess both images
            screen_processed = self.preprocess_image(screen)
            template_processed = self.preprocess_image(template)

            matches = []

            # Template dimensions
            template_height, template_width = template.shape[:2]

            # Try multiple matching methods
            methods = [
                cv2.TM_CCOEFF_NORMED,
                cv2.TM_CCORR_NORMED,
                cv2.TM_SQDIFF_NORMED
            ]

            for method in methods:
                # Template matching
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
                locations = np.where(match_values >= self.confidence)
                
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
                    
                    # Check if point is too close to existing matches
                    if not any(self._is_duplicate((x_center, y_center), existing) for existing in matches):
                        matches.append((x_center, y_center, float(conf)))
                        
                    if not multi_match and matches:
                        break

                if matches and not multi_match:
                    break

            # Sort by confidence
            matches.sort(key=lambda x: x[2], reverse=True)
            
            return matches

        except Exception as e:
            self.logger.error(f"Error finding image: {e}")
            return []

    def _is_duplicate(self, point1, point2, threshold=10):
        """Check if two points are duplicates within threshold"""
        x1, y1 = point1
        x2, y2 = point2[0], point2[1]
        return abs(x1-x2) <= threshold and abs(y1-y2) <= threshold

    def click_image(self, template_path, region=None):
        """Find and click on image"""
        matches = self.find_on_screen(template_path, region)
        if matches:
            x, y, conf = matches[0]
            pyautogui.click(x, y)
            return True
        return False

# Usage example
if __name__ == "__main__":
    finder = ImageFinder(confidence=0.8)

    # loop through list
    search_images = ["halcyon_days.png", "folder_icon.png", "tunic_text.png"]
    #search_images = ["file_notepad.png"]
    #k = "images/koikatsu/"
    #search_images = [k+"btn_add.png",k+"btn_system.png",k+"btn_vnge.png"]
    #search_images = [k+"mmdd_add_asset.png",k+"mmdd_gui.png"]
    #search_images = [k+"effects_checkbox_on.png",k+"elise_checkbox_on.png"]
    #search_images = [k+"system_scene_elise_template_selected.png",k+"system_scene_elise_scene.png"]

    for image in search_images:
        matches = finder.find_on_screen(image)
        if matches:
            x, y, conf = matches[0]
            print(f"Found '{image}' at center ({x}, {y}) with confidence {conf}")
            pyautogui.moveTo(x, y, 1)
            time.sleep(2)
        else:
            print(f"'{image}' not found on screen.")


    # Find single match
    # matches = finder.find_on_screen("halcyon_days.png")
    # if matches:
    #     x, y, conf = matches[0]
    #     print(f"Found at ({x}, {y}) with confidence {conf}")
    
    # Find multiple matches in region
    # region = (100, 100, 500, 500)
    # matches = finder.find_on_screen(
    #     "icon.png", 
    #     region=region,
    #     multi_match=True
    # )

    
    # # Click on image
    # finder.click_image("button.png")