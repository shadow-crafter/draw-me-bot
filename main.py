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


def draw_points():
    pass

def get_edges(img: MatLike) -> MatLike:
    print("Getting edges in image...")

    gray: MatLike = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray: MatLike = cv2.GaussianBlur(gray, (5,5), 1.4)

    v = np.median(gray)
    lower = int(max(0, 0.5 * v))
    upper = int(min(255, 1.2 * v))
    edges = cv2.Canny(gray, lower, upper)
    if cv2.countNonZero(edges) < 100:
        edges = cv2.Canny(gray, max(0, lower // 2), min(255, upper * 1.4))

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

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


def nearest_neighbor_pathfind_segments(points: NDArray, segment_threshold: float = 50.0, min_segment_size=3) -> list[NDArray]:
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
            break

    if current_segment:
        all_segments.append(np.array(current_segment))
    

    return [segment for segment in all_segments if len(segment) >= min_segment_size] #gets rid of noise. like tiny segments are probably not worth drawing


def process_img():
    img: MatLike | None = cv2.imread("test-images\\1.png")
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
        print(f"Found {len(approx_points)} points, {len(sorted_segments)} segments!")

        #show processed image
        traced_image = draw_sorted_segments(img, sorted_segments, draw_numbers=True)
        combined = cv2.hconcat([img, result, traced_image])
        new_scale = (int(combined.shape[1] * 50 / 100), int(combined.shape[0] * 50 / 100))
        combined = cv2.resize(combined, new_scale, interpolation=cv2.INTER_AREA)
        cv2.imshow("Processed Images", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
        time.sleep(1)

        pydirectinput.click()
        pydirectinput.mouseDown(button="left")
        for i, point in enumerate(sorted_points):
            if keyboard.is_pressed('q'):
                break
            pydirectinput.moveTo(point[0], point[1], duration=0.25 + (random.randrange(1, 20) / 100))
            print(f"Point {i}")
        
        pydirectinput.mouseUp(button="left")
    else:
        print("Could not load image!")


def main():
    process_img()


if __name__ == "__main__":
    main()
