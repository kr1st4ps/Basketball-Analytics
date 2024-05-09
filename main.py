"""
Main.
"""

import os
import sys
import cv2
import numpy as np

from utils.court import draw_court_point, get_court_poly, get_keypoints
from utils.functions import (
    draw_flat_points,
    find_frame_transform,
    find_other_court_points,
    is_point_in_frame,
)
from utils.models.y8 import bbox_in_polygon, draw_bboxes, myYOLO
from utils.players import (
    Player,
    bb_intersection_over_union,
    filter_bboxes,
    get_label,
    get_team,
    get_team_coef,
    poly_intersection_over_union,
)

#   Opens video reader
INPUT_VIDEO = "test_video.mp4"
filename, _ = os.path.splitext(INPUT_VIDEO)
court_kp_filename = filename + ".json"
sb_kp_filename = filename + "_sb" + ".json"
cap = cv2.VideoCapture(INPUT_VIDEO)
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
flat_court = cv2.imread("2d_map.png")


#   Opens video
ret, frame = cap.read()
if not ret:
    sys.exit(0)

#   Opens video writer
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)
fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
out_original = cv2.VideoWriter("output.avi", fourcc, fps, frame_size, isColor=True)
out_flat = cv2.VideoWriter(
    "output_flat.avi",
    fourcc,
    fps,
    (flat_court.shape[1], flat_court.shape[0]),
    isColor=True,
)

#   Initializes human detection model
yolo = myYOLO()

#   Initializes SIFT and FLANN algorithms
sift = cv2.SIFT_create(
    nfeatures=0, nOctaveLayers=5, contrastThreshold=0.07, edgeThreshold=50, sigma=1.6
)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict(checks=100))

#   Gets court and scoreboard keypoints
court_keypoints = get_keypoints(court_kp_filename, frame)
scoreboard_keypoints = get_keypoints(sb_kp_filename, frame, "scoreboard")

#   Calculates all other court keypoints
court_keypoints = find_other_court_points(court_keypoints)
court_polygon = get_court_poly(court_keypoints, frame.shape)

#   Gets human bounding boxes in frame
prev_bboxes, curr_bboxes_conf, prev_polys = yolo.detect_persons(frame)
filtered_indices = [
    index
    for index, bbox in enumerate(prev_bboxes)
    if bbox_in_polygon(
        [round(coord) for coord in bbox.cpu().numpy().tolist()], court_polygon
    )
]
prev_bboxes = [prev_bboxes[index] for index in filtered_indices]
curr_bboxes_conf = [curr_bboxes_conf[index] for index in filtered_indices]
prev_polys = [prev_polys[index] for index in filtered_indices]
prev_bboxes, prev_polys, curr_bboxes_conf = filter_bboxes(
    prev_bboxes, prev_polys, curr_bboxes_conf
)

#   Create a class object for each person on court
KMEANS = get_team_coef(prev_bboxes, prev_polys, frame)
players_in_prev_frame = []
for bbox, poly, conf in zip(prev_bboxes, prev_polys, curr_bboxes_conf):
    bbox_rounded = [round(coord) for coord in bbox.cpu().numpy().tolist()]
    if bbox_in_polygon(bbox_rounded, court_polygon):
        c = get_team(bbox_rounded, poly, frame.copy())
        label = get_label(KMEANS, c)
        new_player = Player(1, bbox_rounded, poly, label)
        players_in_prev_frame.append(new_player)

#   Start tracking
prev_frame = frame.copy()
prev_frame_court = frame.copy()
prev_bboxes_court = prev_bboxes.copy()
frame_counter = 1
IOU_THRESHOLD = 0.0
lost_players = []
while True:
    ret, curr_frame = cap.read()
    if not ret:
        break
    if frame_counter % 1 == 0:
        curr_frame_copy = curr_frame.copy()

        #   Gets human bounding boxes in current frame
        curr_bboxes, curr_bboxes_conf, curr_polys = yolo.detect_persons(curr_frame)
        curr_bboxes, curr_polys, curr_bboxes_conf = filter_bboxes(
            curr_bboxes, curr_polys, curr_bboxes_conf
        )

        #   Gets ball and rim detections
        ball_rim_bboxes, ball_rim_conf, ball_rim_classes = (
            yolo.detect_basketball_objects(curr_frame)
        )

    if frame_counter % 5 == 0:
        #   Finds homography matrix between current and previous frame
        H = find_frame_transform(
            prev_frame_court,
            prev_bboxes_court,
            curr_frame,
            curr_bboxes,
            scoreboard_keypoints,
            sift,
            flann,
        )

        #   Extract previous frame points
        prev_frame_points = [
            value for value in court_keypoints.values() if value is not None
        ]
        prev_frame_keys = [
            key for key, value in court_keypoints.items() if value is not None
        ]

        #   Calculate current frame points using the homography matrix
        curr_frame_points = cv2.perspectiveTransform(
            np.array(prev_frame_points, dtype=np.float32).reshape(-1, 1, 2), H
        ).reshape(-1, 2)

        #   Clears dictionary
        for key in court_keypoints:
            court_keypoints[key] = None

        #   Draws points on current frame and sets them in dictionary
        for i, key in enumerate(prev_frame_keys):
            # curr_frame = draw_court_point(curr_frame, curr_frame_points[i], key)

            court_keypoints[key] = (
                round(curr_frame_points[i][0]),
                round(curr_frame_points[i][1]),
            )

        prev_bboxes_court = curr_bboxes.copy()
        prev_frame_court = curr_frame.copy()

    if frame_counter % 1 == 0:
        #   Draw court lines
        court_polygon = get_court_poly(court_keypoints, curr_frame.shape)
        cv2.polylines(
            curr_frame,
            np.int32([court_polygon]),
            isClosed=True,
            color=(255, 0, 0),
            thickness=1,
        )

        found_intersections = []
        players_to_check = [
            player for player in lost_players if frame_counter - player.last_seen < 5
        ] + players_in_prev_frame
        lost_player_ids = [player.id for player in lost_players]
        for bbox, poly, conf in zip(curr_bboxes, curr_polys, curr_bboxes_conf):
            bbox = [round(coord) for coord in bbox.cpu().numpy().tolist()]
            for player in players_to_check:
                iou = bb_intersection_over_union(bbox, player.bbox_history[0])
                if iou > IOU_THRESHOLD:
                    found_intersections.append((player, bbox, poly, iou))

        found_intersections.sort(key=lambda x: x[3], reverse=True)
        intersection_count = {}
        for item in found_intersections:
            key = item[0].id
            intersection_count[key] = intersection_count.get(key, 0) + 1

        found_bboxes = []
        found_players = []
        players_in_frame = []
        for intersection in found_intersections:
            player, bbox, poly, iou = intersection
            if player.id not in found_players and bbox not in found_bboxes:
                if intersection_count[player.id] > 1 and iou < 0.5:
                    c = get_team(bbox, poly, curr_frame_copy.copy())
                    label = get_label(KMEANS, c)
                    if label != player.team:
                        intersection_count[player.id] -= 1
                        continue

                player.update(bbox, poly, frame_counter)

                players_in_frame.append(player)

                found_bboxes.append(bbox)
                found_players.append(player.id)
            intersection_count[player.id] -= 1

        lost_players = [
            player for player in lost_players if player.id not in found_players
        ] + [
            player for player in players_in_prev_frame if player.id not in found_players
        ]

        found_pairs = []
        for bbox, poly, conf in zip(curr_bboxes, curr_polys, curr_bboxes_conf):
            bbox = [round(coord) for coord in bbox.cpu().numpy().tolist()]
            if bbox not in found_bboxes and bbox_in_polygon(bbox, court_polygon):
                for player in lost_players:
                    player_center = np.mean(player.bbox_history[0], axis=0)
                    bbox_center = np.mean(bbox, axis=0)
                    dist = np.linalg.norm(player_center - bbox_center)

                    found_pairs.append((player, bbox, poly, dist))

        found_pairs.sort(key=lambda x: x[3])

        for pair in found_pairs:
            player, bbox, poly, dist = pair
            c = get_team(bbox, poly, curr_frame.copy())
            label = get_label(KMEANS, c)
            if (
                frame_counter - player.last_seen >= 5
                and dist < 150
                and player.team == label
                and player.id not in found_players
                and bbox not in found_bboxes
            ):
                player.update(bbox, poly, frame_counter)

                players_in_frame.append(player)

                found_bboxes.append(bbox)
                found_players.append(player.id)

        lost_players = [
            player for player in lost_players if player.id not in found_players
        ]
        new_players = [
            (bbox, poly, conf)
            for bbox, poly, conf in zip(curr_bboxes, curr_polys, curr_bboxes_conf)
            if [round(coord) for coord in bbox.cpu().numpy().tolist()]
            not in found_bboxes
        ]

        for bbox, poly, conf in new_players:
            bbox = [round(coord) for coord in bbox.cpu().numpy().tolist()]
            if bbox_in_polygon(bbox, court_polygon):
                c = get_team(bbox, poly, curr_frame_copy.copy())
                label = get_label(KMEANS, c)
                new_player = Player(frame_counter, bbox, poly, label)
                players_in_frame.append(new_player)

        # for player in players_in_frame:
        #     if (frame_counter - player.start_frame) % 15 == 0:
        #         player.check_team(curr_frame, KMEANS)

        #   Draw bboxes
        test1 = curr_frame.copy()
        for p in players_in_frame:
            x1, y1, x2, y2 = p.bbox_history[0]
            if p.team == 0:
                cv2.rectangle(test1, (x1, y1), (x2, y2), (255, 0, 0), 2)
            else:
                cv2.rectangle(test1, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                test1,
                f"ID: {p.id}",
                # f"TEAM: {p.team}",
                # f"({x1}, {y1});({x2}, {y2})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            test1,
            f"FRAME: {frame_counter}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

        #   Draw ball and rim
        for bbox, conf, cls in zip(ball_rim_bboxes, ball_rim_conf, ball_rim_classes):
            cls = int(cls)
            bbox = [round(coord) for coord in bbox.cpu().numpy().tolist()]
            if cls == 0:
                color = (0, 165, 255)
            elif cls == 1:
                color = (255, 255, 255)

            cv2.rectangle(test1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        #   Draw players on flat image
        points_in_frame = dict(
            (key, coord)
            for key, coord in court_keypoints.items()
            if is_point_in_frame(coord, curr_frame.shape[1], curr_frame.shape[0])
        )
        flat_court_with_players = draw_flat_points(
            points_in_frame, players_in_frame, flat_court.copy()
        )
        out_flat.write(flat_court_with_players)

        #   Writes frame to video
        out_original.write(test1)
        # out.write(curr_frame)
        # if frame_counter == 500:
        #     sys.exit(0)

        #   Sets current frame and bounding boxes as previous frames to be used for next frame
        prev_bboxes = curr_bboxes
        prev_polys = curr_polys
        prev_frame = curr_frame_copy
        players_in_prev_frame = players_in_frame

    # else:
    #     out.write(curr_frame)
    print(f"FRAMES: {frame_counter}/{TOTAL_FRAMES}")
    frame_counter += 1

#   Closes video and windows
cap.release()
cv2.destroyAllWindows()
out_original.release()
out_flat.release()
