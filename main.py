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
flat_court = cv2.imread(
    "/Users/kristapsalmanis/projects/Basketball-Analytics/2d_map.png"
)

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

#   Gets human bounding boxes in frame
prev_bboxes, curr_bboxes_conf, prev_polys = yolo.get_bboxes(frame)

#   Create a class object for each person on court
TEAM_COEF = get_team_coef(prev_bboxes, prev_polys, frame)
court_polygon = get_court_poly(court_keypoints, frame.shape)
players_in_prev_frame = []
for bbox, poly, conf in zip(prev_bboxes, prev_polys, curr_bboxes_conf):
    if conf > 0.5:
        bbox_rounded = [round(coord) for coord in bbox.numpy().tolist()]
        if bbox_in_polygon(bbox_rounded, court_polygon):
            team_value = get_team(bbox_rounded, poly, frame.copy(), TEAM_COEF)
            new_player = Player(1, bbox_rounded, poly, team_value)
            players_in_prev_frame.append(new_player)

#   Start tracking
prev_frame = frame.copy()
frame_counter = 1
IOU_THRESHOLD = 0.2
lost_players = []
while True:
    ret, curr_frame = cap.read()
    if not ret:
        break
    if frame_counter % 1 == 0:
        curr_frame_copy = curr_frame.copy()

        #   Gets human bounding boxes in current frame
        curr_bboxes, curr_bboxes_conf, curr_polys = yolo.get_bboxes(curr_frame)
        curr_bboxes, curr_bboxes_conf = filter_bboxes(curr_bboxes, curr_bboxes_conf)

        #   Finds homography matrix between current and previous frame
        H = find_frame_transform(
            prev_frame,
            prev_bboxes,
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

        #   Draw court lines
        court_polygon = get_court_poly(court_keypoints, curr_frame.shape)
        cv2.polylines(
            curr_frame,
            np.int32([court_polygon]),
            isClosed=True,
            color=(255, 0, 0),
            thickness=1,
        )

        #   Finds all intersections with previous frame detections
        found_intersections = []
        for player in players_in_prev_frame:
            for bbox, poly, conf in zip(curr_bboxes, curr_polys, curr_bboxes_conf):
                if conf > 0.5:
                    bbox_rounded = [round(coord) for coord in bbox.numpy().tolist()]
                    iou = poly_intersection_over_union(poly, player.poly_history[0])
                    if iou > IOU_THRESHOLD:
                        found_intersections.append((player, bbox_rounded, poly, iou))

        #   Finds all intersections with lost players
        found_matches = []
        for player in lost_players:
            for bbox, poly, conf in zip(curr_bboxes, curr_polys, curr_bboxes_conf):
                if conf > 0.5:
                    bbox_rounded = [round(coord) for coord in bbox.numpy().tolist()]
                    iou = poly_intersection_over_union(poly, player.poly_history[0])
                    if iou > IOU_THRESHOLD:
                        found_intersections.append((player, bbox_rounded, poly, iou))
                        if player.id not in found_matches:
                            found_matches.append(player.id)
        lost_players = [x for x in lost_players if x.id not in found_matches]

        #   Sorts to have pairs with highest IoU in front
        found_intersections.sort(key=lambda x: x[3], reverse=True)

        #   Updates players data
        found_players = []
        found_bboxes = []
        players_in_frame = []
        for intersection in found_intersections:
            player = intersection[0]
            bbox = intersection[1]
            poly = intersection[2]
            if player.id not in found_players and bbox not in found_bboxes:
                found_players.append(player.id)
                found_bboxes.append(bbox)

                player.update(bbox, poly, frame_counter)
                players_in_frame.append(player)

        #   Adds new players
        for g, t, c in zip(curr_polys, curr_bboxes, curr_bboxes_conf):
            b = [round(coord) for coord in t.numpy().tolist()]
            if b not in found_bboxes and c > 0.5 and bbox_in_polygon(b, court_polygon):
                team_value = get_team(b, g, curr_frame.copy(), TEAM_COEF)
                new_player = Player(frame_counter, b, g, team_value)
                players_in_frame.append(new_player)

        #   Keeps track of which players got lost
        for p in players_in_prev_frame:
            if p.id not in found_players:
                lost_players.append(p)

        #   Deletes player object if it was last seen more than 10 frames ago
        lost_players = [x for x in lost_players if frame_counter - x.last_seen < 10]

        #   Draw bboxes
        test1 = curr_frame.copy()
        for p in players_in_frame:
            x1, y1, x2, y2 = p.bbox_history[0]
            if p.team == "home":
                cv2.rectangle(test1, (x1, y1), (x2, y2), (255, 0, 0), 2)
            else:
                cv2.rectangle(test1, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # cv2.putText(
            #     test1,
            #     # f"ID: {p.id}",
            #     f"TEAM: {p.team}",
            #     (x1, y1 - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 0),
            #     1,
            #     cv2.LINE_AA,
            # )

        #   Draw players on flat image
        points_in_frame = dict(
            (key, coord)
            for key, coord in court_keypoints.items()
            if is_point_in_frame(coord, curr_frame.shape[1], curr_frame.shape[0])
        )
        flat_court_with_players = draw_flat_points(
            points_in_frame, players_in_frame + lost_players, flat_court.copy()
        )
        out_flat.write(flat_court_with_players)

        #   Writes frame to video
        out_original.write(test1)
        # out.write(curr_frame)
        if frame_counter == 500:
            sys.exit(0)

        #   Sets current frame and bounding boxes as previous frames to be used for next frame
        prev_bboxes = curr_bboxes
        prev_polys = curr_polys
        prev_frame = curr_frame_copy
        players_in_prev_frame = players_in_frame

    # else:
    #     out.write(curr_frame)

    frame_counter += 1

#   Closes video and windows
cap.release()
cv2.destroyAllWindows()
out_original.release()
out_flat.release()
