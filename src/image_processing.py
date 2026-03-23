import cv2
import numpy as np
from numpy.typing import NDArray
from cv2.typing import MatLike
from src.pathfinding import cleanup_points


def color_quantize(img: MatLike, k: int) -> MatLike:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    data = lab.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, attempts=10, flags=cv2.KMEANS_PP_CENTERS)

    quant = centers[labels.flatten()].reshape(lab.shape).astype(np.uint8)
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    return quant


def get_edges(img: MatLike) -> MatLike:
    print("Getting edges in image...")

    #consider HED or BDCN (?) edge detection
    gray: MatLike = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    #this method gets outer edges well
    hist = cv2.equalizeHist(gray)
    edges_outer = cv2.Canny(hist, 100, 200)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_applied = clahe.apply(gray)
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
