# AutoActionPy
Experimental image-based mouse movement: Finds images onscreen to perform automatic mouse movement, mouse clicks and keyboard presses using a custom actions list

This attempts to make it easier to automate things like clicking on various things onscreen. This will find move the mouse to position onscreen based on images saved in the images folder. can also click mouse and simulate keyboard presses. To make it easier to create images of things you want the mouse to move to, i recommend something like greenshot which can save images of regions on the screen: [https://getgreenshot.org/downloads/](https://getgreenshot.org/downloads/) 
You will probably have to replace the images in the example folder for this to work if your notepad looks different on your computer than the images.

Run these commands in command line to install:
```
python -m venv venv
call venv\Scripts\activate.bat
pip install pyautogui pillow opencv-python keyboard numpy
```
Run with this command to make it do the actions on the action list inside autoaction.py:
```
python autoaction.py
```



Actions list example:
```
#---Opens a new notepad, types in it and saves it to a file named newfile.txt
# Navigate to a menu
run notepad.exe
wait notepad_new.png
click notepad_textarea.png

# Type in notepad
type_text "Hello world!"

# open save window
click menu_file.png menu_save_as.png
click menu_save_as.png btn_save.png

# Type save filename
type_text "hello world.txt"

# Take a full screenshot and clicks on save
screenshot "result.png" full
click btn_save.png
```
