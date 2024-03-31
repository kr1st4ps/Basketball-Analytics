from itertools import combinations
import json
import os
import sys
import cv2
import numpy as np

from utils.constants import *
from utils.d2 import *
from utils.y8 import *
from utils.functions import *

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
}

#   Keypoint tracker parameters
lk_params = dict(winSize=(50, 50),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#   Opens video
full_filename = 'test_video.mp4'
filename, _ = os.path.splitext(full_filename)
start_point_filename = filename + ".json"
cap = cv2.VideoCapture(full_filename)

#   Initializes model
# detectron = myDetectron("small")
yolo = myYOLO()

#       FOR NOW - deprecated
#   Allows user to find a frame on which to put keypoints
# cv2.namedWindow('Frame')
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     cv2.imshow('Frame', frame)
    
#     key = cv2.waitKey(0) & 0xFF
    
#     if key == ord('n'):
#         current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 24)
#         continue

#     elif key == ord('p'):
#         current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 2)
#         continue
        
#     elif key == ord('q'):
#         break
# result = frame.copy()
# cv2.destroyAllWindows()

#   User inputs all visible keypoints using his mouse or starting points are loaded in
ret, frame = cap.read()
if not ret:
    sys.exit(0)
if os.path.exists(start_point_filename):
    with open(start_point_filename, "r") as file:
        frame_court_kp = json.load(file)
else:
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', select_point)
    for key in frame_court_kp:
        while True:
            frame_copy = frame.copy()
            
            for key, point in frame_court_kp.items():
                if point is not None:
                    cv2.circle(frame_copy, point, 5, (255, 0, 0), -1)
                    cv2.putText(frame_copy, key, (point[0], point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            cv2.putText(frame_copy, f"Place {key}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Image', frame_copy)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n'):
                break
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                quit()

    with open(start_point_filename, "w") as file:
        json.dump(frame_court_kp, file)
cv2.destroyAllWindows()

#   Finds coordinates of all points
frame_court_kp, points_in_frame = find_other_court_points(frame_court_kp, frame)

#   Find all persons on the court
    # YOLOv8
outputs = yolo.get_bboxes(frame)
outputs = filter_bboxes_by_polygon(outputs, [frame_court_kp["TOP_LEFT"], frame_court_kp["TOP_RIGHT"], frame_court_kp["BOTTOM_RIGHT"], frame_court_kp["BOTTOM_LEFT"]])
    # detectron2
# outputs = detectron.get_shapes(frame, 0)
# outputs = filter_predictions_by_court(outputs, [frame_court_kp["TOP_LEFT"], frame_court_kp["TOP_RIGHT"], frame_court_kp["BOTTOM_RIGHT"], frame_court_kp["BOTTOM_LEFT"]], frame.shape[:2])

#   Find good court keypoints (for next frame)
keypoints_to_track = get_trackable_points(points_in_frame, outputs, frame.shape[:2])

#   Resets the court coordinate dictionary
for key in frame_court_kp:
    frame_court_kp[key] = None

#   Start tracking
prev_frame = frame.copy()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
end = False
while True:
    #   Takes every 10th frame
    for _ in range(3):
        ret, curr_frame = cap.read()
        if not ret:
            end = True
            break
    if end:
        break

    #   Gets grayscale frame
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    #   Gets new coordinates for tracked points
    tracked_keys = list(keypoints_to_track.keys())
    tracked_coordinates = np.array(list(keypoints_to_track.values())).astype(np.float32).reshape(-1, 1, 2)
    new_coordinates, status, error = cv2.calcOpticalFlowPyrLK(prev_frame_gray, curr_frame_gray, tracked_coordinates, None, **lk_params)
    for i, key in enumerate(tracked_keys):
        frame_court_kp[key] = new_coordinates[i][0]

    #   Finds coordinates of all points
    frame_court_kp, points_in_frame = find_other_court_points(frame_court_kp, curr_frame)

    #   Draws points which are within the frame
    for key, point in points_in_frame.items():
        cv2.circle(curr_frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

    #   Find all persons on the court
        # YOLOv8
    outputs = yolo.get_bboxes(curr_frame)
    outputs = filter_bboxes_by_polygon(outputs, [frame_court_kp["TOP_LEFT"], frame_court_kp["TOP_RIGHT"], frame_court_kp["BOTTOM_RIGHT"], frame_court_kp["BOTTOM_LEFT"]])
        # detectron2
    # outputs = detectron.get_shapes(frame, 0)
    # outputs = filter_predictions_by_court(outputs, [frame_court_kp["TOP_LEFT"], frame_court_kp["TOP_RIGHT"], frame_court_kp["BOTTOM_RIGHT"], frame_court_kp["BOTTOM_LEFT"]], frame.shape[:2])

    #   Draws detections on the frame
        # detectron2
    # annotated_img = detectron.get_annotated_image(curr_frame, outputs)
        # YOLOv8
    annotated_img = draw_bboxes_on_image(curr_frame, outputs)

    #   Show new frame
    cv2.imshow("Tracking", annotated_img)
    cv2.waitKey(1)

    #   Find good court keypoints (for next frame)
    keypoints_to_track = get_trackable_points(points_in_frame, outputs, frame.shape[:2])

    #   TODO - Check if there are at least 4 good points

    #   Resets the court coordinate dictionary
    for key in frame_court_kp:
        frame_court_kp[key] = None

    prev_frame = curr_frame.copy()
    prev_frame_gray = curr_frame_gray.copy()

#   Closes video and windows
cap.release()
cv2.destroyAllWindows()
