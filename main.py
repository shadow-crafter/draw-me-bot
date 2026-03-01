import cv2
import numpy as np

img = cv2.imread("test-images\\1.png")
if img is not None:
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(img_hsv, lower_red, upper_red)
    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("ex", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print("Hello world :D")

"""
-Processing: simplify (and maybe color quantize) -> edge detect -> maybe pathfindind for speed
-calibrate + scaling
-move mouse and have delays.
"""
