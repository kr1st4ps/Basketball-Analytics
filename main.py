"""
Main.
"""

import os
import sys
import cv2
import numpy as np

from utils.court import get_keypoints
from utils.functions import (
    find_homography,
    find_other_court_points,
    is_point_in_frame,
    is_point_in_polygon,
)
from utils.models.y8 import myYOLO

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

# Initializes SIFT and FLANN algorithms
sift = cv2.SIFT_create(
    nfeatures=0, nOctaveLayers=5, contrastThreshold=0.07, edgeThreshold=50, sigma=1.6
)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict(checks=100))

#   Gets court and scoreboard keypoints
court_keypoints = get_keypoints(court_kp_filename, frame)
scoreboard_keypoints = get_keypoints(sb_kp_filename, frame, "scoreboard")

#   Gets human bounding boxes in frame
prev_bboxes = yolo.get_bboxes(frame)

#   Calculates all other court keypoints
court_keypoints = find_other_court_points(court_keypoints, frame)

#   Start tracking
prev_frame = frame.copy()
END = False
while True:
    #   Takes every 10th frame
    for _ in range(10):
        ret, curr_frame = cap.read()
        if not ret:
            END = True
            break
    if END:
        break

    curr_frame_copy = curr_frame.copy()

    #   Gets human bounding boxes in current frame
    curr_bboxes = yolo.get_bboxes(curr_frame)

    #   Finds homography matrix between current and previous frame
    H = find_homography(
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
        cv2.putText(
            curr_frame,
            key,
            (round(curr_frame_points[i][0]), round(curr_frame_points[i][1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(
            curr_frame,
            (round(curr_frame_points[i][0]), round(curr_frame_points[i][1])),
            5,
            (0, 255, 0),
            -1,
        )

        court_keypoints[key] = (
            round(curr_frame_points[i][0]),
            round(curr_frame_points[i][1]),
        )

    #   Draw court lines
    top_left = court_keypoints["TOP_LEFT"]
    top_right = court_keypoints["TOP_RIGHT"]
    if is_point_in_frame(top_left, curr_frame.shape[1], curr_frame.shape[0]):
        bottom_left = court_keypoints["BOTTOM_LEFT"]
    else:
        bottom_left = court_keypoints["BOTTOM_LEFT_HASH"]
    if is_point_in_frame(top_right, curr_frame.shape[1], curr_frame.shape[0]):
        bottom_right = court_keypoints["BOTTOM_RIGHT"]
    else:
        bottom_right = court_keypoints["BOTTOM_RIGHT_HASH"]
    cv2.line(curr_frame, top_left, top_right, (0, 0, 255), thickness=2)
    cv2.line(curr_frame, top_right, bottom_right, (0, 0, 255), thickness=2)
    cv2.line(curr_frame, bottom_right, bottom_left, (0, 0, 255), thickness=2)
    cv2.line(curr_frame, bottom_left, top_left, (0, 0, 255), thickness=2)
    court_polygon = np.array(
        [top_left, top_right, bottom_right, bottom_left], dtype=np.int32
    ).reshape((-1, 1, 2))

    #   Draw detection bounding boxes
    for bbox in curr_bboxes:
        x1, y1, x2, y2 = [round(float(coord)) for coord in bbox]
        if is_point_in_polygon(((x1 + ((x2 - x1) / 2)), y2), court_polygon):
            cv2.rectangle(curr_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    #   Writes frame to video
    out.write(curr_frame)

    #   Sets current frame and bounding boxes as previous frames to be used for next frame
    prev_bboxes = curr_bboxes
    prev_frame = curr_frame_copy

#   Closes video and windows
cap.release()
cv2.destroyAllWindows()
out.release()
