import keyboard
from numpy.typing import NDArray
import pydirectinput
import time


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
