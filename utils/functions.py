"""
Helper functions.
"""

from itertools import combinations
import math
import sys
import cv2
import matplotlib.path as mpltPath
import numpy as np

from utils.constants import LINES, REAL_COURT_KP


def point_to_flat(bbox, h, img_shp):
    """
    Gets point coordinates on flat image.
    """
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


def draw_images(keypoint_dict, players, frame, flat_court):
    """
    Draws on normal and flat images.
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

    h_to_flat, _ = cv2.findHomography(np.array(frame_img), np.array(flat_img))
    h_to_frame, _ = cv2.findHomography(np.array(flat_img), np.array(frame_img))

    #   Finds all player flat court coordinates normalized
    coords = []
    for player in players:
        bbox = player.bbox_history[0]
        flat_pos_tuple = point_to_flat(bbox, h_to_flat, flat_court.shape)
        if len(player.bbox_history) > 1:
            prev_bbox = player.bbox_history[1]
            prev_flat_pos_tuple = point_to_flat(prev_bbox, h_to_flat, flat_court.shape)
            player.update_distance(
                distance_between_points(
                    flat_pos_tuple[0],
                    flat_pos_tuple[1],
                    prev_flat_pos_tuple[0],
                    prev_flat_pos_tuple[1],
                )
            )

        coords.append(
            [
                flat_pos_tuple[0] / flat_court.shape[1],
                flat_pos_tuple[1] / flat_court.shape[0],
            ]
        )

    #   Draws each player
    clean_frame = frame.copy()
    players_and_flat_points = zip(players, coords)
    players_and_flat_points = sorted(players_and_flat_points, key=lambda x: x[1][1])
    print("len", len(players_and_flat_points))
    for player, p in players_and_flat_points:
        print(len(player.poly_history))
        print(player.poly_history[0])
    for player, flat_point in players_and_flat_points:
        print("start iter")
        if player.team == 1:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        print(1)
        flat_point_s = [
            round(flat_point[0] * flat_court.shape[1]),
            round(flat_point[1] * flat_court.shape[0]),
        ]
        print(2)
        flat_point_l = [
            round(flat_point[0] * 2800),
            round(flat_point[1] * 1500),
        ]
        print(3)

        overlay = np.zeros((1500, 2800, 3), np.uint8)
        print(4)
        cv2.circle(overlay, flat_point_l, 40, color, -1)
        print(5)

        cv2.circle(flat_court, flat_point_s, 20, color, thickness=-1)
        print(6)

        if player.has_ball:
            print(7)
            cv2.circle(overlay, flat_point_l, 45, (0, 165, 255), 2)
            print(8)
            cv2.circle(flat_court, flat_point_s, 25, (0, 165, 255), 2)
            print(9)

        print(10)
        if player.number != "":
            print(11)
            cv2.putText(
                overlay,
                str(player.number),
                (
                    flat_point_l[0] - (len(str(player.number)) * 10),
                    flat_point_l[1] + 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            print(12)
            shift = 5 if len(str(player.number)) == 1 else 15
            print(13)
            cv2.putText(
                flat_court,
                str(player.number),
                (flat_point_s[0] - shift, flat_point_s[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            print(14)

        print(15)
        #   Draws circle on ground
        overlay = cv2.warpPerspective(overlay, h_to_frame, (1280, 720))
        print(16)
        gray_image = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        print(17)
        _, mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        print(18)
        mask_3channels = cv2.merge((mask, mask, mask))
        print(19)
        frame[mask_3channels > 0] = 0
        print(20)
        frame += overlay * (mask_3channels > 0)
        print(21)

        #   Draws polygon over the circle
        if len(player.poly_history[0]) > 0:
            poly_mask = np.zeros_like(frame[:, :, 0])
            print(22)
            poly = np.array([player.poly_history[0]], dtype=np.int32)
            print(poly)
            print(23)
            cv2.fillPoly(poly_mask, poly, color=255)
            print(24)
            mask_3channels = cv2.merge((poly_mask, poly_mask, poly_mask))
            print(25)
            frame[mask_3channels > 0] = 0
            print(26)
            frame += clean_frame * (mask_3channels > 0)
        print("done with iter")

    print("done")
    return frame, flat_court, players


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
        x, y, w, h = np.vectorize(round)(np.vectorize(float)(bbox.cpu()))
        cv2.rectangle(mask_old, (x, y), (w, h), 0, -1)
    for bbox in new_bboxes:
        x, y, w, h = np.vectorize(round)(np.vectorize(float)(bbox.cpu()))
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


def bbox_intersect(bbox1, bbox2):
    """
    Checks if 2 bboxes intersect at all
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    return (
        (x1_max >= x2_min)
        and (x2_max >= x1_min)
        and (y1_max >= y2_min)
        and (y2_max >= y1_min)
    )


def distance_between_points(x1, y1, x2, y2):
    """
    Gets distance between 2 points.
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def round_bbox(bbox):
    """
    Rounds bbox values.
    """
    return [round(coord) for coord in bbox.cpu().numpy().tolist()]


def bb_intersection_over_union(bbox_a, bbox_b):
    """
    Finds IoU (intersection over union) value of two bboxes.
    """
    intersection_width = max(
        0, min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0]) + 1
    )
    intersection_height = max(
        0, min(bbox_a[3], bbox_b[3]) - max(bbox_a[1], bbox_b[1]) + 1
    )
    intersection_area = intersection_width * intersection_height

    bbox_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    bbox_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    iou = intersection_area / float(bbox_a_area + bbox_b_area - intersection_area)

    return iou
