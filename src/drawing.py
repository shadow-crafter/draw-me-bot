import keyboard
from numpy.typing import NDArray
import pydirectinput
import time
import tkinter as tk

class AreaSelector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes('-alpha', 0.3)
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.config(cursor="cross")

        self.canvas = tk.Canvas(self.root, cursor="cross", bg="grey")
        self.canvas.pack(fill="both", expand=True)

        self.start_pos: list[int] = [None, None]
        self.rect: int = None
        self.selection: tuple | None = None

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.root.bind("<Escape>", lambda _: self.root.destroy())

        self.canvas.focus_set()
    
    def on_button_press(self, event):
        self.canvas.delete("all")
        self.start_pos[0] = event.x
        self.start_pos[1] = event.y
        self.rect = self.canvas.create_rectangle(self.start_pos[0], self.start_pos[1], 1, 1, outline="red", width=2)
    
    def on_move(self, event):
        self.canvas.coords(self.rect, self.start_pos[0], self.start_pos[1], event.x, event.y) #gets the rectangle and updates it
    
    def on_button_release(self, event):
        x1, x2 = sorted([self.start_pos[0], event.x]) #in case someone draws a "negative" rectangle
        y1, y2 = sorted([self.start_pos[1], event.y])

        self.selection = (x1, y1, x2 - x1, y2 - y1)

        self.canvas.itemconfigure(self.rect, fill="green", stipple="gray50", outline="green")
    
    def get_selection(self):
        self.root.mainloop()
        return self.selection


def draw_points(segments: list[NDArray]) -> None:
    time.sleep(1)

    pydirectinput.moveTo(segments[0][0][0], segments[0][0][1], duration=0.1)
    pydirectinput.click()
    for i, segment in enumerate(segments):
        print(f"Segment {i}")
        for j, point in enumerate(segment):
            if keyboard.is_pressed('q'):
                return
            print(f"Point {j}: {point}")
            if j == 0:
                pydirectinput.mouseUp(button="left")
                pydirectinput.moveTo(point[0], point[1], duration=0.1)
                pydirectinput.mouseDown(button="left")
            pydirectinput.moveTo(point[0], point[1], duration=0.15)
        pydirectinput.mouseUp(button="left")
