import cv2
from cv2.typing import MatLike
import src.drawing as drawing
import src.image_processing as image_processing
from src.pathfinding import nearest_neighbor_pathfind_segments


def process_img():
    img: MatLike | None = cv2.imread("test-images\\3.png")
    if img is not None:
        edges: MatLike = None
        try:
            mode = int(input("What mode (1 = smoothed, 2 = quantized, 3 = raw): "))
            if mode == 1:
                smoothed = cv2.pyrMeanShiftFiltering(img, sp=16, sr=32)
                edges = image_processing.get_edges(smoothed)
            elif mode == 2:
                quant = image_processing.color_quantize(img, k=13)
                edges = image_processing.get_edges(quant)
            else: #assume raw
                edges = image_processing.get_edges(img)
        except Exception as e:
            print(f"Trouble processing image: {e}")
            return

        result, approx_points = image_processing.get_points_approx(edges)
        sorted_segments = nearest_neighbor_pathfind_segments(approx_points)
        print(f"Found {len(approx_points)} points originally, {sum(map(len, sorted_segments))} points sorted, and {len(sorted_segments)} segments!")

        #show processed image
        traced_image = image_processing.draw_sorted_segments(img, sorted_segments, draw_numbers=True)
        combined = cv2.hconcat([img, result, traced_image])
        new_scale = (int(combined.shape[1] * 50 / 100), int(combined.shape[0] * 50 / 100))
        combined = cv2.resize(combined, new_scale, interpolation=cv2.INTER_AREA)
        cv2.imshow("Processed Images", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        drawing.draw_points(sorted_segments)
    else:
        print("Could not load image!")


def main():
    process_img()


if __name__ == "__main__":
    main()
