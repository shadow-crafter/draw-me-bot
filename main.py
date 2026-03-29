import cv2
from cv2.typing import MatLike
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import pyautogui
import src.drawing as drawing
import src.image_processing as image_processing
from src.pathfinding import nearest_neighbor_pathfind_segments, offset_points_in_segments


def get_base_image() -> MatLike | None:
    img : MatLike | None = None
    Path("logs/").mkdir(parents=True, exist_ok=True)
    try:
        reuse = 2 # assume 2 if there is not already a screenshot in logs
        if Path("logs/region_screenshot.png").exists():
            reuse = int(input("Would you like to use the previous screenshot (1) or take a new one (2)? : "))
        
        if reuse == 1:
            img = cv2.imread("logs/region_screenshot.png")
        elif reuse == 2:
            input("Press enter when you are ready to take screenshot.")
            selector = drawing.AreaSelector()
            region = selector.get_selection()
            if not region:
                print("Could not get region from area selection.")
            
            screenshot = pyautogui.screenshot("logs/region_screenshot.png", region=region)
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Invalid response. Expected 1 or 2, got: {reuse}")
    except Exception as e:
        print(f"Trouble getting screenshot data: {e}")
        return
    
    return img


def process_img(img: MatLike) -> MatLike | None:
    edges: MatLike | None = None
    try:
        mode = int(input("What mode (1 = smoothed, 2 = quantized, 3 = raw)? : "))
        if mode == 1:
            smoothed = cv2.pyrMeanShiftFiltering(img, sp=16, sr=32)
            edges = image_processing.get_edges(smoothed)
        elif mode == 2:
            quant = image_processing.color_quantize(img, k=13)
            edges = image_processing.get_edges(quant)
        elif mode == 3:
            edges = image_processing.get_edges(img)
        else:
            raise ValueError(f"Invalid mode provided. Expected 1, 2, or 3, got: {mode}")
    except Exception as e:
        print(f"Trouble processing image: {e}")
        return
    
    return edges


def show_processed_img(img: MatLike, result: MatLike, sorted_segments: list[NDArray]):
    traced_image = image_processing.draw_sorted_segments(img, sorted_segments, draw_numbers=True)
    combined = cv2.hconcat([img, result, traced_image])
    new_scale = (int(combined.shape[1] * 50 / 100), int(combined.shape[0] * 50 / 100))
    combined = cv2.resize(combined, new_scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite("logs/processed_images.png", combined)
    cv2.imshow("Processed Images", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_points(img: MatLike, sorted_segments: list[NDArray]):
    input("Press enter when you are ready to select where to draw.")
    height, width = img.shape[:2]
    visualizer = drawing.DrawVisualizer((width, height), sorted_segments)
    corner_pos = visualizer.get_corner_pos()
    if not corner_pos:
        print("Could not get corner position.")
        return
    offset_segments = offset_points_in_segments(sorted_segments, corner_pos)

    delay: bool = True if input("Add delay to draw input (Y or N): ").lower() == 'y' else False
    drawing.draw_points(offset_segments, delay)


def main():
    img: MatLike | None = get_base_image()
    if img is None:
        print("Could not load image!")
        return
    
    edges: MatLike | None = process_img(img)
    if edges is None:
        print("Could not process image.")
        return

    # get point data from edges
    result, approx_points = image_processing.get_points_approx(edges)
    if len(approx_points) <= 5:
        print("Not enough points found in image.")
        return
    
    sorted_segments: list[NDArray] = nearest_neighbor_pathfind_segments(approx_points)
    print(f"Found {len(approx_points)} points originally, {sum(map(len, sorted_segments))} points sorted, and {len(sorted_segments)} segments!")
    
    show_processed_img(img, result, sorted_segments)
    draw_points(img, sorted_segments)


if __name__ == "__main__":
    main()
