"""
Helper functions.
"""

from itertools import combinations
import sys
import cv2
import matplotlib.path as mpltPath
import numpy as np

from .constants import *


def find_point(point, h):
    """
    Finds point using homography.
    """
    point_orig = np.array(point)
    point_image2_homogeneous = np.dot(h, point_orig)
    point_image2_normalized = point_image2_homogeneous / point_image2_homogeneous[2]
    point_image2_euclidean = (point_image2_normalized[0], point_image2_normalized[1])

    return (round(point_image2_euclidean[0][0]), round(point_image2_euclidean[1][0]))


def point_to_flat(bbox, h, img_shp):
    feet_pos = np.array(
        [[(bbox[0] + ((bbox[2] - bbox[0]) / 2), bbox[3])]], dtype=np.float32
    )
    flat_pos = cv2.perspectiveTransform(feet_pos, h)

    flat_pos_tuple = (
        int(flat_pos[0][0][0] / 2800 * img_shp[1]),
        int(flat_pos[0][0][1] / 1500 * img_shp[0]),
    )

    return flat_pos_tuple


def area_of_polygon(vertices):
    """
    Calculates area of a polygon.
    """
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    area = abs(area) / 2.0
    return area


def find_largest_polygon(coords_dict):
    """
    Finds largest polygon from user given points.
    """
    existing_points = [
        (key, coord) for key, coord in coords_dict.items() if coord is not None
    ]

    max_area = 0
    largest_polygon = None
    for i in range(4, len(existing_points) + 1):
        for combo in combinations(existing_points, i):
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


def find_other_court_points(keypoint_dict):
    """
    Calculates coordinates of all other court keypoints.
    """
    largest_polygon = find_largest_polygon(keypoint_dict)
    if largest_polygon is None:
        print("Not enough points")
        sys.exit(0)

    flat_img = []
    frame_img = []
    for point in largest_polygon:
        frame_img.append([point[1][0], point[1][1]])
        flat_img.append(REAL_COURT_KP[point[0]])

    h_to_frame, _ = cv2.findHomography(np.array(flat_img), np.array(frame_img))

    keys_with_none_values = [
        key for key, value in keypoint_dict.items() if value is None
    ]
    for key in keys_with_none_values:
        keypoint_dict[key] = find_point(
            [[REAL_COURT_KP[key][0]], [REAL_COURT_KP[key][1]], [1]], h_to_frame
        )

    return keypoint_dict


def draw_flat_points(keypoint_dict, players, img):
    """ """
    largest_polygon = find_largest_polygon(keypoint_dict)
    if largest_polygon is None:
        print("Not enough points")
        sys.exit(0)

    flat_img = []
    frame_img = []
    for point in largest_polygon:
        frame_img.append([point[1][0], point[1][1]])
        flat_img.append(REAL_COURT_KP[point[0]])

    h_to_flat, _ = cv2.findHomography(np.array(frame_img), np.array(flat_img))

    for player in players:
        for i in range(len(player.bbox_history) - 1):
            first_point = point_to_flat(player.bbox_history[i], h_to_flat, img.shape)
            second_point = point_to_flat(
                player.bbox_history[i + 1], h_to_flat, img.shape
            )
            cv2.line(
                img,
                first_point,
                second_point,
                (0, 255, 0),
                thickness=2,
            )

        bbox = player.bbox_history[0]
        flat_pos_tuple = point_to_flat(bbox, h_to_flat, img.shape)

        cv2.circle(img, flat_pos_tuple, 20, (255, 0, 0), thickness=-1)
        shift = 5 if len(str(player.id)) == 1 else 15
        text_position = (flat_pos_tuple[0] - shift, flat_pos_tuple[1] + 10)
        cv2.putText(
            img,
            str(player.id),
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            thickness=2,
        )

    return img


def find_frame_transform(
    old_frame, old_bboxes, new_frame, new_bboxes, scoreboard, sift, flann
):
    """
    Finds homography matrix between two frames.
    """
    height, width = old_frame.shape[:2]
    mask_old = np.ones((height, width), dtype=np.uint8) * 255
    mask_new = np.ones((height, width), dtype=np.uint8) * 255

    for bbox in old_bboxes:
        x, y, w, h = np.vectorize(round)(np.vectorize(float)(bbox))
        cv2.rectangle(mask_old, (x, y), (w, h), 0, -1)
    for bbox in new_bboxes:
        x, y, w, h = np.vectorize(round)(np.vectorize(float)(bbox))
        cv2.rectangle(mask_new, (x, y), (w, h), 0, -1)

    sb = mpltPath.Path(np.array([val for _, val in scoreboard.items()])).vertices
    sb_tl, _, sb_br, _ = sb
    cv2.rectangle(
        mask_old,
        (round(sb_tl[0]), round(sb_tl[1])),
        (round(sb_br[0]), round(sb_br[1])),
        0,
        -1,
    )
    cv2.rectangle(
        mask_new,
        (round(sb_tl[0]), round(sb_tl[1])),
        (round(sb_br[0]), round(sb_br[1])),
        0,
        -1,
    )

    kp1, des1 = sift.detectAndCompute(old_frame, mask_old)
    kp2, des2 = sift.detectAndCompute(new_frame, mask_old)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return h
    else:
        return None


def is_point_in_frame(point, frame_width, frame_height):
    """
    Check if a point is within the frame boundaries.
    """
    x, y = point
    return 0 <= x < frame_width and 0 <= y < frame_height
