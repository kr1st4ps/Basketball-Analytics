from itertools import combinations
import sys
import cv2
import numpy as np

from .constants import *
from .d2 import are_bboxes_intersect

#   Gets ROI around a passed point
def get_roi_bbox_around_point(image_size, point, width, height):
    x, y = point
    half_width = width // 2
    half_height = height // 2
    x_min = max(0, x - half_width)
    y_min = max(0, y - half_height)
    x_max = min(image_size[1], x + half_width)
    y_max = min(image_size[0], y + half_height)
    return x_min, y_min, x_max, y_max

#   Finds point using homography
def find_point(point, H):
    point_orig = np.array(point)
    point_image2_homogeneous = np.dot(H, point_orig)
    point_image2_normalized = point_image2_homogeneous / point_image2_homogeneous[2]
    point_image2_euclidean = (point_image2_normalized[0], point_image2_normalized[1])

    return (int(point_image2_euclidean[0][0]), int(point_image2_euclidean[1][0]))

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
    order_of_keys = None

    for i in range(4, len(valid_coords) + 1):
        for combo in combinations(valid_coords, i):
            vertices = [coord for _, coord in combo]
            area = area_of_polygon(vertices)
            if area > max_area:
                max_area = area
                largest_polygon = combo
                order_of_keys = [key for key, _ in combo]

    if largest_polygon is None:
        return None, None, None
    else:
        return largest_polygon, max_area, order_of_keys
    
#   Checks if passed point is within the frames borders
def is_point_inside_frame(point, frame_size):
    x, y = point
    height, width = frame_size
    return 0 <= x < width and 0 <= y < height

#   Saves coordinates of user selected point
def select_point(event, x, y, flags, params):
    global frame_court_kp, key
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_court_kp[key] = (x, y)

#   Calculates coordinates of other court keypoints
def find_other_court_points(kp_dict, frame):
    #   Finds largest polygon
    largest_polygon, max_area, order_of_keys = largest_polygon_area(kp_dict)
    if largest_polygon is None:
        print("Not enough points") #TODO try again
        sys.exit(0)

    #   Collects respective points from the flat court image
    flat_img = []
    frame_img = []
    for point in largest_polygon:
        frame_img.append([point[1][0], point[1][1]])
        flat_img.append(REAL_COURT_KP[point[0]])
    
    #   Generates homography matrixes both ways
    H_to_flat, _ = cv2.findHomography(np.array(frame_img), np.array(flat_img))
    H_to_frame, _ = cv2.findHomography(np.array(flat_img), np.array(frame_img))
    
    #   Finds coordinates of other points TODO - rewrite this section
    keys_with_none_values = [key for key, value in kp_dict.items() if value is None]
    points_in_frame = {key: value for key, value in kp_dict.items() if value is not None and is_point_inside_frame(kp_dict[key], frame.shape[:2])}
    for key in keys_with_none_values:
        kp_dict[key] = find_point([[REAL_COURT_KP[key][0]], [REAL_COURT_KP[key][1]], [1]], H_to_frame)

        if is_point_inside_frame(kp_dict[key], frame.shape[:2]):
            points_in_frame[key] = kp_dict[key]

    return kp_dict, points_in_frame


#   Find good court keypoints (for next frame)
def get_trackable_points(points_in_frame, outputs, frame_size):
    keypoints_to_track = {}
    for key, point in points_in_frame.items():
        point_roi_bbox = get_roi_bbox_around_point(frame_size, point, 30, 30)
        
        is_good_point = True

            # detectron2
        # for bbox in outputs["instances"].pred_boxes:
            # YOLOv8
        for bbox in outputs:
            if are_bboxes_intersect(point_roi_bbox, bbox):
                is_good_point = False
                break

        if is_good_point:
            keypoints_to_track[key] = point

    return keypoints_to_track