"""
Main.
"""

import os
import sys
import cv2
import numpy as np

from utils.court import draw_court_point, get_court_poly, get_keypoints
from utils.functions import (
    find_frame_transform,
    find_other_court_points,
)
from utils.models.y8 import bbox_in_polygon, draw_bboxes, myYOLO
from utils.players import Player, bb_intersection_over_union

#   Opens video reader
INPUT_VIDEO = "test_video.mp4"
filename, _ = os.path.splitext(INPUT_VIDEO)
court_kp_filename = filename + ".json"
sb_kp_filename = filename + "_sb" + ".json"
cap = cv2.VideoCapture(INPUT_VIDEO)

#   Opens video
ret, frame = cap.read()
if not ret:
    sys.exit(0)

#   Opens video writer
fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
out = cv2.VideoWriter("output.avi", fourcc, 6.0, (1280, 720), isColor=True)

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
prev_bboxes = yolo.get_bboxes(frame)

#   Create a class object for each person on court
court_polygon = get_court_poly(court_keypoints, frame.shape)
players_in_prev_frame = []
for bbox in prev_bboxes:
    bbox_rounded = [round(coord) for coord in bbox.numpy().tolist()]
    if bbox_in_polygon(bbox_rounded, court_polygon):
        new_player = Player(1, bbox_rounded)
        players_in_prev_frame.append(new_player)

test1 = frame.copy()
for p in players_in_prev_frame:
    x1, y1, x2, y2 = p.bbox_history[0]
    cv2.rectangle(test1, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(
        test1,
        f"ID: {p.id}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
cv2.imwrite("test1.png", test1)

#   Start tracking
prev_frame = frame.copy()
frame_counter = 1
IOU_THRESHOLD = 0.2
while True:
    ret, curr_frame = cap.read()
    if not ret:
        break
    if frame_counter % 1 == 0:
        curr_frame_copy = curr_frame.copy()

        #   Gets human bounding boxes in current frame
        curr_bboxes = yolo.get_bboxes(curr_frame)

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
            curr_frame, court_polygon, isClosed=True, color=(255, 0, 0), thickness=3
        )

        #   Check if any of the current bboxes intersect with previously found bboxes
        players_in_frame = []
        players_found = []
        for bbox in curr_bboxes:
            bbox_rounded = [round(coord) for coord in bbox.numpy().tolist()]
            new = True
            for player in players_in_prev_frame:
                if (
                    bb_intersection_over_union(bbox_rounded, player.bbox_history[0])
                    > IOU_THRESHOLD
                    and player.id not in players_found
                ):
                    player.update(bbox_rounded)
                    players_found.append(player.id)
                    players_in_frame.append(player)
                    new = False
                    break
            if new and bbox_in_polygon(bbox_rounded, court_polygon):
                new_player = Player(frame_counter, bbox_rounded)
                players_in_frame.append(new_player)

        # if frame_counter == 1000:
        test1 = curr_frame.copy()
        for p in players_in_frame:
            x1, y1, x2, y2 = p.bbox_history[0]
            cv2.rectangle(test1, (x1, y1), (x2, y2), (255, 0, 0), 2)
            print(f"DRAWING ID: {p.id}")
            cv2.putText(
                test1,
                f"ID: {p.id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        # cv2.imwrite("test2.png", test1)

        # sys.exit(0)
        #   Draw detection bounding boxes
        # curr_frame = draw_bboxes(curr_frame, curr_bboxes, court_polygon)

        #   Writes frame to video
        out.write(test1)
        # out.write(curr_frame)
        if frame_counter == 1000:
            sys.exit(0)

        #   Sets current frame and bounding boxes as previous frames to be used for next frame
        prev_bboxes = curr_bboxes
        prev_frame = curr_frame_copy
        players_in_prev_frame = players_in_frame

    # else:
    #     out.write(curr_frame)

    frame_counter += 1

#   Closes video and windows
cap.release()
cv2.destroyAllWindows()
out.release()
