# AutoActionPy - Experimental image-based automation

This attempts to make it easier to automate tasks like clicking on various things onscreen. AutoActionPy is a experimental Python-based automation tool that lets you create image-based mouse and keyboard automation scripts. It finds images on your screen and performs actions like mouse movements, clicks, and keyboard inputs automatically. You can create a list of actions (macros) in a text file that you wish it to perform.
You will probably have to replace the images in the example folder for this to work if your notepad looks different on your computer than the images.

## 🌟 Features

- **Image-Based Automation**: Finds and interacts with UI elements using image recognition
- **Mouse Actions**: Move, click, double-click, right-click, and drag operations
- **Keyboard Input**: Type text and simulate keyboard shortcuts
- **Flexible Commands**: Wait for images on-screen, take screenshots, run programs
- **Emergency Stop**: Press 'ESC' to stop automation at any time
- **Conditional Actions**: Execute actions based on presence/absence of images
- **Dual Search Modes**: Normal (precise) and Experimental (more tolerant) image matching
- **Manual Region Selection**: Shift-triggered manual region selection for more precise automation
- **Action Recording**: Record mouse clicks and keyboard inputs to generate automation scripts
- **Duplicate Detection**: Smart handling of repeated images during recording
- **Enhanced Mouse Actions**: Support for middle-click and relative mouse movements
- **Advanced Keyboard Support**: Handling of key combinations and special keys

## 🚀 Installation

Run these commands in command line to install:
```
python -m venv venv
call venv\Scripts\activate.bat
pip install pyautogui pillow opencv-python keyboard numpy
```

## 📝 Usage

To run the example:
```
python autoaction.py
```

To run with your own action file:  
  1. Create an action list file (e.g., my_actions.txt) with your automation commands
  2. Run this command in command line:
```
python autoaction.py my_actions.txt
```  
  - You can also set the folder to look for images in:
```
python autoaction.py my_actions.txt --images_path path/to/images/folder
```

## Example Action List
```text
#---Opens a new notepad, types in it and saves as text file
# Navigate to a menu
run notepad.exe
wait notepad_new.png
click notepad_textarea.png

# Type in notepad
type "Hello world!"

# open save window
click menu_file.png
click menu_save_as.png

# Type save filename
type "hello world.txt"

# Take a full screenshot and saves file
screenshot "result.png"
click btn_save.png
```

## Available Commands

- `click [image]` - Move to and Left click on image, use current position if no image given
- `click_right [image]` - Move to and Right click on image, use current position if no image given
- `click_middle [image]` - Move to and Middle click on image, use current position if no image given
- `double_click [image]` - Move to and Double click (with left mouse button) on image, use current position if no image given
- `wait <image> [timeout]` - Wait for image to appear
- `wait <seconds>` - Wait for specified seconds
- `type "text"` - Type text (supports {key} combinations)
- `move_to <image>` - Move mouse to image
- `move_to <x> <y>` - Move mouse to x and/or y position, use +/- to make relative like `move_to +10 -5`
- `move_between <image1> <image2> [percentage]` - Move to point between two images
- `drag_to <image>` - Drag from current position to image
- `drag_between <image1> <image2> [percentage]` - Drag between two images
- `search_mode normal|experimental` - Set image matching precision
- `jump_to <image>` - Instantly move to image
- `press_key <key>` - Press a keyboard key
- `screenshot "savename.png"` - Take screenshot
- `screenshot_region "savename.png"` - Take screenshot of last region (take full screenshot if not available)
- `run <program>` - Launch a program

## 🎯 Conditional Commands

Actions can be made conditional by adding `if` and `if_not` parameters to allow for more dynamic automation scripts:
- `click image.png if other_image.png` - Only click if the image other_image.png is present
- `click image.png if_not blocker.png` - Only click if the image blocker.png is NOT present
- `type "text" if prompt.png` - Only type if prompt is visible

## 🎥 Recording Action Lists Automatically
`record_demonstration.py` is a tool that lets you create action lists by recording your actions while also taking cropped screenshots. 
It is kind of like a "smart recorder" that watches what you do and turns it into a script that AutoActionPy (autoaction.py) can replay later.
Type what you want recorded normally, but for mouse clicks to be captured, press either CTRL (automatic cropped screenshot) or SHIFT (manual screenshot region selecting) to simulate left mouse clicks.
To record your own automation:
Recording controls:
- F9: Start/Stop recording
- F10: Pause/Resume recording
- CTRL: Capture click with automatic region detection
- SHIFT: Capture click with manual region selection
- ESC: Emergency stop

The recorder will generate:
- An actions.txt file with your recorded commands
- PNG images of the clicked regions in the images folder

Note: 
- Currently this only captures keyboard key presses, waiting, and left mouse clicks (through pressing either CTRL or SHIFT so simulate mouse click), as well as generate image assets.
- Actual mouse clicks isn't captured by record_demonstration.py (to help avoid errors), so press CTRL or SHIFT when you want to simulate a mouse click.

## 📸 Creating Image Assets from Stratch
`record_demonstration.py` can make creation of image assets far easier, but if you perfer to make them manually instead, follow these steps:  
1. Use a screenshot tool to capture the UI elements/things you want the mouse to move to:
   - Run `python screenshot_region.py` then press shift to take a screenshot, then drag the mouse to make a rectangle region. This screenshot region will be saved as a png.
   - Alternately, use a screenshot tool like [Greenshot](https://getgreenshot.org/downloads/) which can also save regions of the screen as png files.
   - Alternately, press printscreen to take full screenshots on your pc, then open something like Paint, then paste and crop the image. Save as a png file.
2. Save images in a folder inside the project directory.
3. Reference images in your action list by filename.

## 💡 Tips

- `search_mode` is always normal by default and is recommended, but you can try `search_mode experimental` which tries mutliple search methods but is more likely to fail/get false positives
- Manual region selection (SHIFT) provides more precise targeting
- Key combinations can be typed using {key} format: {ctrl+c}, {shift+tab}
- Wait commands support multiple images: `wait image1.png or image2.png`
- Mouse movements support relative coordinates: `move_to "x+100" "y-50"`
- Use high-quality, distinctive screenshots (save as lossless PNG files) for reliable recognition
- Test your automation scripts with the emergency stop key (ESC) ready
- Start with simple actions and build up to more complex sequences
- Adjust timing using wait commands if needed
