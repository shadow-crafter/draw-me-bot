import cv2
import numpy as np
from numpy.typing import NDArray
from cv2.typing import MatLike
import keyboard
import pydirectinput
import random
from scipy.spatial import KDTree
import time


def color_quantize(img: MatLike, k: int) -> MatLike:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    data = lab.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, attempts=10, flags=cv2.KMEANS_PP_CENTERS)

    quant = centers[labels.flatten()].reshape(lab.shape).astype(np.uint8)
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    return quant


def cleanup_points(pts, min_dist=5):
    out = []
    for p in pts:
        if all(abs(p[0] - q[0]) > min_dist or abs(p[1] - q[1]) > min_dist for q in out):
            out.append(p)
    return out


def draw_sorted_segments(img: MatLike, segments: list[NDArray], draw_numbers: bool = True) -> MatLike:
    result = img.copy()
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for seg_idx, segment in enumerate(segments):
        color = colors[seg_idx % len(colors)]
        
        # Draw lines between consecutive points
        for i in range(len(segment) - 1):
            pt1 = tuple(segment[i].astype(int))
            pt2 = tuple(segment[i + 1].astype(int))
            cv2.line(result, pt1, pt2, color, 2)
        
        # Draw point circles and optional numbers
        for point_idx, point in enumerate(segment):
            pt = tuple(point.astype(int))
            cv2.circle(result, pt, 6, color, -1)  # Filled circle
            
            if draw_numbers and point_idx % 5 == 0: #only show numbers every 5
                cv2.putText(result, str(point_idx), (pt[0] + 10, pt[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return result


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
    

def get_edges(img: MatLike) -> MatLike:
    print("Getting edges in image...")

    #consider HED or BDCN (?) edge detection
    gray: MatLike = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    #this method gets outer edges well
    hist = cv2.equalizeHist(gray)
    edges_outer = cv2.Canny(hist, 100, 200)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_applied = clahe.apply(gray)
    #gray = cv2.bilateralFilter(gray, 7, 50, 50)
    edges_inner = cv2.Canny(clahe_applied, 50, 150)

    combined_edges = cv2.bitwise_or(edges_outer, edges_inner)

    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)

    return edges


def get_points_approx(img: MatLike) -> tuple[MatLike, NDArray]:
    print("Getting points on edges...")

    new_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    point_list = []

    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour) #filter out small noise
        if area < 150:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 35:
            continue

        #epsilon = 0.005 * perimeter if perimeter > 800 else 0.02 * perimeter
        epsilon = max(0.005, min(0.03, 0.01 * perimeter))
        approx = cv2.approxPolyDP(contour, epsilon, True)

        hull = cv2.convexHull(approx)
        cv2.polylines(new_img, [hull], True, (0, 255, 0), 1)

        for point in approx:
            x, y = point[0]
            cv2.circle(new_img, (x, y), 1, (0, 0, 255), -1)
            point_list.append((int(x), int(y)))
    
    point_list = cleanup_points(point_list, min_dist=6)
    point_list = np.array(point_list)
    
    return new_img, point_list


def nearest_neighbor_pathfind_segments(points: NDArray, segment_threshold: float = 25.0, min_segment_size=1) -> list[NDArray]:
    print("Sorting points...")

    tree = KDTree(points)

    unvisited = set(range(len(points)))
    curr_i = 0

    all_segments: list[NDArray] = []
    current_segment = [points[curr_i]]
    unvisited.remove(curr_i)

    while unvisited:
        k = min(len(unvisited) + 1, len(points))
        distances, indexes = tree.query(points[curr_i], k=k)

        found_next = False
        for dist, i in zip(distances, indexes):
            i = int(i)
            if i not in unvisited:
                continue
            
            if dist > segment_threshold: #big jump between closest point, so create new segment
                all_segments.append(np.array(current_segment))
                current_segment = []
            
            current_segment.append(points[i])
            unvisited.remove(i)
            curr_i = i
            found_next = True
            break
        
        if not found_next:
            #save current segment and find the nearest unvisited point to start a new segment
            if current_segment:
                all_segments.append(np.array(current_segment))
            
            if unvisited:
                nearest_unvisited = min(unvisited, key=lambda idx: np.linalg.norm(points[idx] - points[curr_i]))
                curr_i = nearest_unvisited
                current_segment = [points[curr_i]]
                unvisited.remove(curr_i)

    if current_segment:
        all_segments.append(np.array(current_segment))
    

    return [segment for segment in all_segments if len(segment) >= min_segment_size] #gets rid of noise. tiny segments are probably not worth drawing


def process_img():
    img: MatLike | None = cv2.imread("test-images\\3.png")
    if img is not None:
        edges: MatLike = None
        try:
            mode = int(input("What mode (1 = smoothed, 2 = quantized, 3 = raw): "))
            if mode == 1:
                smoothed = cv2.pyrMeanShiftFiltering(img, sp=16, sr=32)
                edges = get_edges(smoothed)
            elif mode == 2:
                quant = color_quantize(img, k=13)
                edges = get_edges(quant)
            else: #assume raw
                edges = get_edges(img)
        except Exception as e:
            print(f"Trouble processing image: {e}")
            return

        result, approx_points = get_points_approx(edges)
        sorted_segments = nearest_neighbor_pathfind_segments(approx_points)
        print(f"Found {len(approx_points)} points originally, {sum(map(len, sorted_segments))} points sorted, and {len(sorted_segments)} segments!")

        #show processed image
        traced_image = draw_sorted_segments(img, sorted_segments, draw_numbers=True)
        combined = cv2.hconcat([img, result, traced_image])
        new_scale = (int(combined.shape[1] * 50 / 100), int(combined.shape[0] * 50 / 100))
        combined = cv2.resize(combined, new_scale, interpolation=cv2.INTER_AREA)
        cv2.imshow("Processed Images", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        draw_points(sorted_segments)
    else:
        print("Could not load image!")


def main():
    process_img()


if __name__ == "__main__":
    main()
