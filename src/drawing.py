import keyboard
from numpy.typing import NDArray
import pydirectinput
import random
import tkinter as tk


class AreaSelector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes("-alpha", 0.3)
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
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
        self.root.bind("<Return>", lambda _: self.root.destroy())

        self.canvas.create_text(
            self.root.winfo_screenwidth() / 2,
            self.root.winfo_screenheight() / 2,
            text="Click and drag to select area",
            font=("Arial", 16, "bold"),
        )

        self.canvas.focus_set()

    def on_button_press(self, event):
        self.canvas.delete("all")

        self.start_pos[0] = event.x
        self.start_pos[1] = event.y
        self.rect = self.canvas.create_rectangle(
            self.start_pos[0],
            self.start_pos[1],
            1,
            1,
            outline="red",
            width=2,
            dash=(3, 20),
        )

    def on_move(self, event):
        self.canvas.coords(
            self.rect, self.start_pos[0], self.start_pos[1], event.x, event.y
        )  # gets the rectangle and updates it

    def on_button_release(self, event):
        x1, x2 = sorted(
            [self.start_pos[0], event.x]
        )  # in case someone draws a "negative" rectangle
        y1, y2 = sorted([self.start_pos[1], event.y])

        self.selection = (x1, y1, x2 - x1, y2 - y1)

        self.canvas.itemconfigure(
            self.rect, fill="green", stipple="gray50", outline="green"
        )
        self.canvas.create_text(
            x1,
            y1,
            text="Press ESCAPE or RETURN to confirm selection.",
            font=("Arial", 12, "bold"),
            anchor="sw",
        )

    def get_selection(self):
        self.root.mainloop()
        return self.selection


class DrawVisualizer:
    def __init__(self, img_size: tuple, segments: list[NDArray]):
        self.img_size = img_size
        self.segments = segments
        
        self.root = tk.Tk()
        self.root.attributes("-alpha", 0.3)
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.config(cursor="cross")

        self.canvas = tk.Canvas(self.root, cursor="cross", bg="grey")
        self.canvas.pack(fill="both", expand=True)

        self.corner_pos: tuple | None = None

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)

        self.root.bind("<Escape>", lambda _: self.root.destroy())
        self.root.bind("<Return>", lambda _: self.root.destroy())

        self.canvas.create_text(
            self.root.winfo_screenwidth() / 2,
            self.root.winfo_screenheight() / 2,
            text="Click to choose the top-left corner to draw from",
            font=("Arial", 16, "bold"),
        )

        self.canvas.focus_set()

    def on_button_press(self, event):
        self.canvas.delete("all")

        self.corner_pos = (event.x, event.y)
        self.rect = self.canvas.create_rectangle(
            self.corner_pos[0],
            self.corner_pos[1],
            self.corner_pos[0] + self.img_size[0],
            self.corner_pos[1] + self.img_size[1],
            outline="red",
            width=4,
            dash=(3, 20),
        )

        colors = ["blue", "green", "yellow", "purple", "orange"]
        for segment in self.segments:
            for i, point in enumerate(segment):
                pos = (point[0] + self.corner_pos[0], point[1] + self.corner_pos[1])
                self.canvas.create_oval(pos[0] - 1, pos[1] - 1, pos[0] + 1, pos[1] + 1, fill=colors[i % len(colors)])
        
        self.canvas.create_text(
            self.corner_pos[0],
            self.corner_pos[1],
            text="Press ESCAPE or RETURN to confirm selection.",
            font=("Arial", 12, "bold"),
            anchor="sw",
        )

    def get_corner_pos(self):
        self.root.mainloop()
        return self.corner_pos


def draw_points(segments: list[NDArray], delay: bool) -> None:
    draw_time_est = 0.125 * sum(map(len, segments)) if delay else 0.05 * len(segments)
    print(f"Estimated draw time: {draw_time_est:.2f} seconds (delay: {delay}). Hold 'q' during the drawing process to cancel at any time.")
    input("Press enter to start drawing.")

    if not delay:
        pydirectinput.PAUSE = 0
    pydirectinput.moveTo(segments[0][0][0], segments[0][0][1], duration=0.1)
    pydirectinput.click()
    for i, segment in enumerate(segments):
        print(f"Segment {i}")
        for j, (x, y) in enumerate(segment):
            if keyboard.is_pressed("q"):
                pydirectinput.mouseUp(button="left")
                return

            print(f"Point {j}: ({x}, {y})")

            if j == 0:
                pydirectinput.mouseUp(button="left")
                pydirectinput.moveTo(x, y)
                pydirectinput.mouseDown(button="left")
            else:
                pydirectinput.moveTo(x, y, duration = 0.05 + (random.random() / 10))
        
        pydirectinput.mouseUp(button="left")
