import json
import os
import sys
import cv2
import numpy as np

from utils.constants import *
from utils.d2 import *
from utils.y8 import *
from utils.functions import *

#   Saves coordinates of user selected court keypoint
def select_point_kp(event, x, y, flags, params):
    global frame_court_kp, name
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_court_kp[name] = (x, y)

#   Saves coordinates of user selected scoreboard keypoint
def select_point_sb(event, x, y, flags, params):
    global scoreboard, name
    if event == cv2.EVENT_LBUTTONDOWN:
        scoreboard[name] = (x, y)

#   Dictionary for storing keypoint coordinates in the frame
frame_court_kp = {
    "TOP_LEFT": None, 
    "TOP_LEFT_HASH": None,
    "TOP_MID": None,
    "TOP_RIGHT_HASH": None,
    "TOP_RIGHT": None,

    "RIGHT_FT_TOP_RIGHT": None,
    "RIGHT_FT_TOP_LEFT": None,
    "RIGHT_FT_BOTTOM_LEFT": None,
    "RIGHT_FT_BOTTOM_RIGHT": None,

    "BOTTOM_RIGHT": None,
    "BOTTOM_RIGHT_HASH": None,
    "BOTTOM_MID": None,
    "BOTTOM_LEFT_HASH": None,
    "BOTTOM_LEFT": None,
    
    "LEFT_FT_BOTTOM_LEFT": None,
    "LEFT_FT_BOTTOM_RIGHT": None,
    "LEFT_FT_TOP_RIGHT": None,
    "LEFT_FT_TOP_LEFT": None,
    
    "CENTER_TOP": None,
    "CENTER_BOTTOM": None,

    "VB_TOP_LEFT": None,
    "VB_TOP_LEFT_MID": None,
    "VB_TOP_MID": None,
    "VB_TOP_RIGHT_MID": None,
    "VB_TOP_RIGHT": None,

    "VB_BOTTOM_RIGHT": None,
    "VB_BOTTOM_RIGHT_MID": None,
    "VB_BOTTOM_MID": None,
    "VB_BOTTOM_LEFT": None,
    "VB_BOTTOM_LEFT_MID": None,
}

scoreboard = {
    "TL": None,
    "TR": None,
    "BR": None,
    "BL": None,
}

#   Opens video reader
full_filename = 'test_video2.mp4'
filename, _ = os.path.splitext(full_filename)
start_point_filename = filename + ".json"
sb_filename = filename + "_sb" + ".json"
cap = cv2.VideoCapture(full_filename)

#   Opens video writer
ret, frame = cap.read()
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # Use 'x264' if 'mp4v' doesn't work
out = cv2.VideoWriter('v_largest_asdspoly.avi', fourcc, 6.0, (1280, 720), isColor=True)

#   Initializes model
yolo = myYOLO()

# Initializes SIFT and FLANN algorithms
sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.07, edgeThreshold=50, sigma=1.6)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict(checks=100))

#   User inputs all visible keypoints using his mouse or starting points are loaded in
if not ret:
    sys.exit(0)
if os.path.exists(start_point_filename):
    with open(start_point_filename, "r") as file:
        frame_court_kp = json.load(file)
else:
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', select_point_kp)
    for name in frame_court_kp:
        while True:
            frame_copy = frame.copy()
            
            for name1, point in frame_court_kp.items():
                if point is not None:
                    cv2.circle(frame_copy, point, 5, (255, 0, 0), -1)
                    cv2.putText(frame_copy, name1, (point[0], point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            cv2.putText(frame_copy, f"Place {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Image', frame_copy)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n'):
                break
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                quit()
    cv2.destroyAllWindows()

    with open(start_point_filename, "w") as file:
        json.dump(frame_court_kp, file)

#   User inputs all scoreboard corners using his mouse or points are loaded in
if os.path.exists(sb_filename):
    with open(sb_filename, "r") as file:
        scoreboard = json.load(file)
else:
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', select_point_sb)
    for name in scoreboard:
        while True:
            frame_copy = frame.copy()
            
            for name1, point in scoreboard.items():
                if point is not None:
                    cv2.circle(frame_copy, point, 5, (255, 0, 0), -1)
                    cv2.putText(frame_copy, name1, (point[0], point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            cv2.putText(frame_copy, f"Place {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Image', frame_copy)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n'):
                break
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                quit()
    cv2.destroyAllWindows()

    with open(sb_filename, "w") as file:
        json.dump(scoreboard, file)
cv2.destroyAllWindows()

#   Gets human bounding boxes in frame
prev_bboxes = yolo.get_bboxes(frame)

#   Calculates all other court keypoints
frame_court_kp = find_other_court_points(frame_court_kp, frame)

#   Start tracking
prev_frame = frame.copy()
end = False
while True:
    #   Takes every 10th frame
    for _ in range(10):
        ret, curr_frame = cap.read()
        if not ret:
            end = True
            break
    if end:
        break

    curr_frame_copy = curr_frame.copy()

    #   Gets human bounding boxes in current frame
    curr_bboxes = yolo.get_bboxes(curr_frame)

    #   Finds homography matrix between current and previous frame
    H = find_homography(prev_frame, prev_bboxes, curr_frame, curr_bboxes, scoreboard, sift, flann)

    #   Extract previous frame points
    prev_frame_points = [value for value in frame_court_kp.values() if value is not None]
    prev_frame_keys = [key for key, value in frame_court_kp.items() if value is not None]

    #   Calculate current frame points using the homography matrix
    curr_frame_points = cv2.perspectiveTransform(np.array(prev_frame_points, dtype=np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)

    #   Clears dictionary
    for key in frame_court_kp:
        frame_court_kp[key] = None

    #   Draws points on current frame and sets them in dictionary
    for i, key in enumerate(prev_frame_keys):
        cv2.putText(curr_frame, key, (round(curr_frame_points[i][0]), round(curr_frame_points[i][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(curr_frame, (round(curr_frame_points[i][0]), round(curr_frame_points[i][1])), 5, (0, 255, 0), -1)

        frame_court_kp[key] = (round(curr_frame_points[i][0]), round(curr_frame_points[i][1]))

    #   Draws court borders
    cv2.line(curr_frame, frame_court_kp["TOP_LEFT"], frame_court_kp["TOP_RIGHT"], (0, 0, 255), thickness=2)
    cv2.line(curr_frame, frame_court_kp["TOP_RIGHT"], frame_court_kp["BOTTOM_RIGHT"], (0, 0, 255), thickness=2)
    cv2.line(curr_frame, frame_court_kp["BOTTOM_LEFT"], frame_court_kp["BOTTOM_MID"], (0, 0, 255), thickness=2)
    cv2.line(curr_frame, frame_court_kp["BOTTOM_MID"], frame_court_kp["BOTTOM_RIGHT"], (0, 0, 255), thickness=2)
    cv2.line(curr_frame, frame_court_kp["BOTTOM_LEFT"], frame_court_kp["TOP_LEFT"], (0, 0, 255), thickness=2)

    #   Writes frame to video
    out.write(curr_frame)

    #   Sets current frame and bounding boxes as previous frames to be used for next frame
    prev_bboxes = curr_bboxes
    prev_frame = curr_frame_copy

#   Closes video and windows
cap.release()
cv2.destroyAllWindows()
out.release()