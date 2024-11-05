# AutoActionPy - Experimental image-based automation

This attempts to make it easier to automate tasks like clicking on various things onscreen. AutoActionPy is a experimental Python-based automation tool that lets you create image-based mouse and keyboard automation scripts. It finds images on your screen and performs actions like mouse movements, clicks, and keyboard inputs automatically. You can create a list of actions in a text file that you wish it to perform.
You will probably have to replace the images in the example folder for this to work if your notepad looks different on your computer than the images.

## 🌟 Features

- **Image-Based Automation**: Finds and interacts with UI elements using image recognition
- **Mouse Actions**: Move, click, double-click, right-click, and drag operations
- **Keyboard Input**: Type text and simulate keyboard shortcuts
- **Flexible Commands**: Wait for images on-screen, take screenshots, run programs
- **Emergency Stop**: Press 'ESC' to stop automation at any time
- **Conditional Actions**: Execute actions based on presence/absence of images


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
  2b. You can also set the folder to look for images in:
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
type_text "Hello world!"

# open save window
click menu_file.png
click menu_save_as.png

# Type save filename
type_text "hello world.txt"

# Take a full screenshot and clicks on save
screenshot "result.png" full
click btn_save.png
```

## Available Commands

-  run <program> - Launch a program
-  wait <image> - Wait for image to appear
-  click <image> - Click on image
-  type_text "text" - Type text
-  screenshot "filename.png" [full] - Take screenshot
-  move_to <image> - Move mouse to image
-  drag_to <image> - Drag from current position to image
-  and more.

## 📸 Creating Image Assets

1. Use a screenshot tool like [Greenshot](https://getgreenshot.org/downloads/) which can save regions of the screen as png files.Capture the UI elements/things you want the mouse to move to.
2. Save images in the images folder.
3. Reference images in your action list by filename.

## ⚠️ Tips

- Use high-quality, distinctive screenshots (save as lossless PNG files) for reliable recognition
- Test your automation scripts with the emergency stop key (ESC) ready
- Start with simple actions and build up to more complex sequences
- Adjust timing using wait commands if needed

