import cv2
import numpy as np
from cv2.typing import MatLike


def color_quantize(img: MatLike, k: int) -> MatLike:
    data = img.reshape((-1, 3)).astype(np.float32)

    criteria: tuple = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res: MatLike = center[label.flatten()]
    result: MatLike = res.reshape((img.shape))

    return result


def get_edges(img: MatLike) -> MatLike:
    gray: MatLike = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred: MatLike = cv2.medianBlur(gray, 5)
    edges: MatLike = cv2.Canny(blurred, 100, 500)

    return edges


def process_img():
    img: MatLike | None = cv2.imread("test-images\\1.png")
    if img is not None:
        quant = color_quantize(img, k=16)
        edges = get_edges(quant)

        cv2.imshow("edges detected", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Could not load image!")


def main():
    print("Hello world :D")
    process_img()


if __name__ == "__main__":
    main()
"""
-Processing: simplify (and maybe color quantize) ✅ -> edge detect ✅ -> maybe pathfindind for speed
-calibrate + scaling
-move mouse and have delays.
"""
