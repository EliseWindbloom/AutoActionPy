import keyboard
import pyautogui
from tkinter import *
from PIL import Image, ImageEnhance, ImageTk
import os
from datetime import datetime
import sys
import signal
import time

class ScreenshotTool:
    def __init__(self):
        self.root = None
        self.canvas = None
        self.screenshot = None
        self.start_x = None
        self.start_y = None
        self.current_rectangle = None
        self.original_screenshot = None
        self.is_running = True
        self.keyboard_handler = None
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            self.keyboard_handler = keyboard.on_press_key("shift", self.on_shift_press)
            print("Screenshot tool running. Press Shift to take a screenshot.")
            print("Press Ctrl+Q to quit the program.")
            
            keyboard.add_hotkey('ctrl+q', self.quit_program)
            
        except Exception as e:
            print(f"Error starting keyboard listener: {e}")
            self.quit_program()
    
    def signal_handler(self, signum, frame):
        print("\nReceived shutdown signal. Cleaning up...")
        self.quit_program()
    
    def quit_program(self):
        print("Shutting down screenshot tool...")
        self.is_running = False
        
        if self.keyboard_handler:
            keyboard.unhook_all()
        
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass
        
        sys.exit(0)
        
    def on_shift_press(self, e):
        """Handle Shift key press"""
        if not self.is_running or self.root:  # Prevent multiple windows
            return
            
        try:
            # Take screenshot immediately, before creating any windows
            self.original_screenshot = pyautogui.screenshot()
            
            # Small delay to ensure screenshot is complete before window creation
            time.sleep(0.1)
            
            # Now create the window
            self.create_selection_window()
            
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            if self.root:
                self.root.destroy()
    
    def create_selection_window(self):
        try:
            self.root = Tk()
            
            # Create and configure the window without immediately showing it
            self.root.attributes('-fullscreen', True)
            #self.root.attributes('-alpha', 0.3) #warning, alpha will cause it to blend with desktop, making it look like it's in a half state
            self.root.attributes('-alpha', 1.0)  # Fully opaque
            self.root.withdraw()  # Hide window initially

            # Apply a gray overlay for the GUI version
            gray_overlay = Image.new('RGBA', self.original_screenshot.size, (128, 128, 128, 10))  # RGBA with slight transparency
            tinted_screenshot = Image.alpha_composite(self.original_screenshot.convert('RGBA'), gray_overlay)
            
            # Convert screenshot for tkinter
            self.screenshot = ImageTk.PhotoImage(tinted_screenshot)
            
            self.canvas = Canvas(
                self.root,
                width=self.root.winfo_screenwidth(),
                height=self.root.winfo_screenheight(),
                highlightthickness=0
            )
            self.canvas.pack()
            
            self.canvas.create_image(0, 0, image=self.screenshot, anchor=NW)
            
            # Bind mouse events
            self.canvas.bind("<Button-1>", self.on_mouse_down)
            self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
            
            # Bind Escape key to close current window
            self.root.bind("<Escape>", self.close_window)
            
            # Now show the window and set it topmost
            self.root.deiconify()
            self.root.attributes('-topmost', True)
            self.root.focus_force()
            
            self.root.mainloop()
            
        except Exception as e:
            print(f"Error creating window: {e}")
            self.close_window()
    
    def close_window(self, event=None):
        if self.root:
            try:
                self.root.destroy()
                self.root = None
            except:
                pass
        
        self.start_x = None
        self.start_y = None
        self.current_rectangle = None
        self.original_screenshot = None
        
    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        
        if self.current_rectangle:
            self.canvas.delete(self.current_rectangle)
        self.current_rectangle = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline='blue', width=2
        )
        
    def on_mouse_drag(self, event):
        if self.current_rectangle:
            self.canvas.coords(
                self.current_rectangle,
                self.start_x, self.start_y,
                event.x, event.y
            )
            
    def on_mouse_up(self, event):
        if self.start_x is None or self.start_y is None:
            return
            
        try:
            x1 = min(self.start_x, event.x)
            y1 = min(self.start_y, event.y)
            x2 = max(self.start_x, event.x)
            y2 = max(self.start_y, event.y)
            
            region_screenshot = self.original_screenshot.crop((x1, y1, x2, y2))
            
            if not os.path.exists('screenshots'):
                os.makedirs('screenshots')
                
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'screenshots/screenshot_{timestamp}.png'
            region_screenshot.save(filename)
            
            print(f"Screenshot saved as {filename}")
            
        except Exception as e:
            print(f"Error saving screenshot: {e}")
        finally:
            self.close_window()

if __name__ == "__main__":
    try:
        tool = ScreenshotTool()
        
        while tool.is_running:
            try:
                keyboard.wait()
            except KeyboardInterrupt:
                tool.quit_program()
                break
            
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)