import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree


def cleanup_points(pts, min_dist=5):
    out = []
    for p in pts:
        if all(abs(p[0] - q[0]) > min_dist or abs(p[1] - q[1]) > min_dist for q in out):
            out.append(p)
    return out


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
