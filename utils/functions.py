from itertools import combinations
import sys
import cv2
import matplotlib.path as mpltPath
import numpy as np

from .constants import *
from .d2 import are_bboxes_intersect

#   Finds point using homography
def find_point(point, H):
    point_orig = np.array(point)
    point_image2_homogeneous = np.dot(H, point_orig)
    point_image2_normalized = point_image2_homogeneous / point_image2_homogeneous[2]
    point_image2_euclidean = (point_image2_normalized[0], point_image2_normalized[1])

    return (round(point_image2_euclidean[0][0]), round(point_image2_euclidean[1][0]))


#   Calculates area of a polygon
def area_of_polygon(vertices):
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    area = abs(area) / 2.0
    return area


#   Finds largest polygon from user given points
def largest_polygon_area(coords_dict):
    valid_coords = [(key, coord) for key, coord in coords_dict.items() if coord is not None]

    max_area = 0
    largest_polygon = None
    for i in range(4, len(valid_coords) + 1):
        for combo in combinations(valid_coords, i):
            for _, line in LINES.items():
                points_on_line = []
                for p in combo:
                    if p[0] in line:
                        points_on_line.append(p)
                if len(points_on_line) > 2:
                    for name in points_on_line[1:-1]:
                        combo = tuple(item for item in combo if item[0] != name[0])
                
            if len(combo) < 4:
                continue

            vertices = [coord for _, coord in combo]
            area = area_of_polygon(vertices)
            if area > max_area:
                max_area = area
                largest_polygon = combo

    if largest_polygon is None:
        return None
    else:
        return largest_polygon
    

#   Calculates coordinates of other court keypoints
def find_other_court_points(kp_dict, frame):
    largest_polygon = largest_polygon_area(kp_dict)
    if largest_polygon is None:
        print("Not enough points") #TODO try again
        sys.exit(0)

    flat_img = []
    frame_img = []
    for point in largest_polygon:
        frame_img.append([point[1][0], point[1][1]])
        flat_img.append(REAL_COURT_KP[point[0]])
    
    # H_to_flat, _ = cv2.findHomography(np.array(frame_img), np.array(flat_img))
    H_to_frame, _ = cv2.findHomography(np.array(flat_img), np.array(frame_img))

    # flat_frame = cv2.warpPerspective(frame, H_to_flat, (2800, 1500))
    # for _, v in REAL_COURT_KP.items():
    #     cv2.circle(flat_frame, (round(v[0]), round(v[1])), 5, (0, 255, 0), -1)
    # cv2.imwrite("flat_test.png", flat_frame)
    
    keys_with_none_values = [key for key, value in kp_dict.items() if value is None]
    for key in keys_with_none_values:
        kp_dict[key] = find_point([[REAL_COURT_KP[key][0]], [REAL_COURT_KP[key][1]], [1]], H_to_frame)

    return kp_dict


#   Finds homography matrix between two frames
def find_homography(old_frame, old_bboxes, new_frame, new_bboxes, scoreboard, sift, flann):
    height, width = old_frame.shape[:2]
    mask1 = np.ones((height, width), dtype=np.uint8) * 255
    mask2 = np.ones((height, width), dtype=np.uint8) * 255

    for bbox in old_bboxes:
        x, y, w, h = np.vectorize(round)(np.vectorize(float)(bbox))
        cv2.rectangle(mask1, (x, y), (w, h), 0, -1)
    for bbox in new_bboxes: 
        x, y, w, h = np.vectorize(round)(np.vectorize(float)(bbox))
        cv2.rectangle(mask2, (x, y), (w, h), 0, -1)

    sb = mpltPath.Path(np.array([val for _, val in scoreboard.items()])).vertices
    a, _, c, _ = sb
    cv2.rectangle(mask1, (round(a[0]), round(a[1])), (round(c[0]), round(c[1])), 0, -1)
    cv2.rectangle(mask2, (round(a[0]), round(a[1])), (round(c[0]), round(c[1])), 0, -1)

    kp1, des1 = sift.detectAndCompute(old_frame, mask1)
    kp2, des2 = sift.detectAndCompute(new_frame, mask1)

    matches = flann.knnMatch(des1, des2, k=2)

    # Filter matches using the Lowe ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return H
    else:
        return None
    
