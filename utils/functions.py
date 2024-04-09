from itertools import combinations
import sys
import cv2
import matplotlib.path as mpltPath
import numpy as np

from .constants import *

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

    # cv2.imwrite("frame.png", new_frame)
    # cv2.imwrite("mask.png", mask2)
    kp1, des1 = sift.detectAndCompute(old_frame, mask1)
    # print(kp1)
    # print()
    # print(des1)
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
    
def do_polygons_intersect(poly1, poly2):
    # Convert polygons to a format suitable for OpenCV (if they are not already)
    poly1 = np.array(poly1, dtype=np.int32).reshape((-1, 1, 2))
    poly2 = np.array(poly2, dtype=np.int32).reshape((-1, 1, 2))

    # Create empty images to draw polygons
    image1 = np.zeros((1000, 1000), dtype=np.uint8) # Adjust size as needed
    image2 = np.zeros((1000, 1000), dtype=np.uint8) # Adjust size as needed

    # Draw filled polygons
    cv2.fillPoly(image1, [poly1], 255)
    cv2.fillPoly(image2, [poly2], 255)

    # Check for intersection
    intersection = cv2.bitwise_and(image1, image2)

    # If there's any non-zero pixel in intersection, polygons intersect
    return np.any(intersection)


def clip_point(point, img_shape):
    """Clip the point to the image boundaries."""
    print(img_shape)
    x, y = point
    x = max(0, min(x, img_shape[1] - 1))
    y = max(0, min(y, img_shape[0] - 1))
    return x, y


def is_point_in_polygon(point, polygon_points):
    """
    Check if a point is inside a given polygon.

    Parameters:
    point (tuple): The point to check, given as (x, y).
    polygon_points (list): List of points defining the polygon.

    Returns:
    bool: True if the point is inside the polygon, False otherwise.
    """
    # Ensure all polygon points are tuples
    # for p in polygon_points:
    #     print(p)
    #     print(p[0])
    #     print(p[0][0])
    #     print()
    polygon_points = [(p[0][0], p[0][1]) for p in polygon_points]
    min_x = min([p[0] for p in polygon_points])
    min_y = min([p[1] for p in polygon_points])
    # print(f"X: {min_x}, Y: {min_y}")
    if min_x < 0:
        add_x = abs(min_x)
    else:
        add_x = 0

    if min_y < 0:
        add_y = abs(min_y)
    else:
        add_y = 0

    adjusted_polygon = [(p[0] + add_x, p[1] + add_y) for p in polygon_points]
    adjusted_point = (point[0] + add_x, point[1] + add_y)
    # sys.exit(0)

    # Check if the point is inside the polygon
    # print(adjusted_point)
    return cv2.pointPolygonTest(np.array(adjusted_polygon), adjusted_point, False) >= 0
    # return None

def pad_and_annotate_image(image, pad_size, points_dict):
    """
    Pad an image and annotate it with points.

    Parameters:
    image (numpy.ndarray): The image to pad and annotate.
    pad_size (int): The size of the padding to add to each side of the image.
    points_dict (dict): A dictionary of points to annotate, in the format {'label': (x, y), ...}.

    Returns:
    numpy.ndarray: The padded and annotated image.
    """
    # Pad the image
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Draw each point on the padded image
    for label, point in points_dict.items():
        adjusted_point = (point[0] + pad_size, point[1] + pad_size)
        cv2.circle(padded_image, adjusted_point, radius=5, color=(0, 255, 0), thickness=-1)  # Green point
        cv2.putText(padded_image, label, (adjusted_point[0] + 10, adjusted_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return padded_image


def adjust_sign_if_needed(old_number, new_number):
    """
    Adjusts the sign of the new number if it has a different sign than the old number 
    and its absolute value is greater than 50,000.

    Parameters:
    old_number (float or int): The old number.
    new_number (float or int): The new number to be possibly adjusted.

    Returns:
    float or int: The adjusted new number.
    """
    # Check if signs are different and the absolute value of the new number is greater than 50,000
    if ((old_number * new_number < 0) and (abs(new_number) > 50000)) or abs(new_number-old_number) > 50000:
        # Change the sign of the new number
        return -new_number
    else:
        # Return the new number as it is
        return new_number
    
def is_point_in_frame(point, frame_width, frame_height):
    """
    Check if a point is within the frame boundaries.

    Parameters:
    point (tuple): The point, given as (x, y).
    frame_width (int): Width of the frame.
    frame_height (int): Height of the frame.

    Returns:
    bool: True if the point is inside the frame, False otherwise.
    """
    x, y = point
    return 0 <= x < frame_width and 0 <= y < frame_height