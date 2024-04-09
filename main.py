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
from utils.models.y8 import draw_bboxes, myYOLO

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

#   Gets human bounding boxes in frame
prev_bboxes = yolo.get_bboxes(frame)

#   Calculates all other court keypoints
court_keypoints = find_other_court_points(court_keypoints)

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
        curr_frame = draw_court_point(curr_frame, curr_frame_points[i], key)

        court_keypoints[key] = (
            round(curr_frame_points[i][0]),
            round(curr_frame_points[i][1]),
        )

    #   Draw court lines
    court_polygon = get_court_poly(court_keypoints, curr_frame.shape)
    cv2.polylines(
        curr_frame, court_polygon, isClosed=True, color=(255, 0, 0), thickness=3
    )

    #   Draw detection bounding boxes
    curr_frame = draw_bboxes(curr_frame, curr_bboxes, court_polygon)

    #   Writes frame to video
    out.write(curr_frame)

    #   Sets current frame and bounding boxes as previous frames to be used for next frame
    prev_bboxes = curr_bboxes
    prev_frame = curr_frame_copy

#   Closes video and windows
cap.release()
cv2.destroyAllWindows()
out.release()
