"""
Main.
"""

import os
import sys
import time
import cv2
import numpy as np
import easyocr


from utils.ball_rim import find_game_ball
from utils.court import get_court_poly, get_keypoints, find_other_court_points
from utils.functions import (
    draw_images,
    find_frame_transform,
    round_bbox,
    is_point_in_frame,
)
from utils.models.y8 import bbox_in_polygon, filter_bboxes, myYOLO
from utils.players import (
    Player,
    create_result_json,
    get_team,
    generate_cluster,
    track_players,
)

#   Constants
KP_FOLDER = os.path.join("resources", "coordinates")
VIDEO_FOLDER = os.path.join("resources", "videos")
RESULT_PATH = os.path.join("resources", "runs")
FRAME_COUNTER = 1

#   Opens video
INPUT_VIDEO = os.path.join(VIDEO_FOLDER, "test_video3.mp4")
filename = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
court_kp_file_path = os.path.join(KP_FOLDER, filename + ".json")
sb_kp_file_path = os.path.join(KP_FOLDER, filename + "_sb" + ".json")
result_file_path = os.path.join(RESULT_PATH, "output_" + filename + ".avi")
result_flat_file_path = os.path.join(RESULT_PATH, "output_flat_" + filename + ".avi")
result_data_file_path = os.path.join(RESULT_PATH, "output_" + filename + ".json")

cap = cv2.VideoCapture(INPUT_VIDEO)
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
flat_court = cv2.imread("2d_court.png")

ret, frame = cap.read()
if not ret:
    sys.exit(0)

#   Opens video writer
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size_original = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)
frame_size_flat = (flat_court.shape[1], flat_court.shape[0])
fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
out_original = cv2.VideoWriter(
    result_file_path, fourcc, fps, frame_size_original, isColor=True
)
out_flat = cv2.VideoWriter(
    result_flat_file_path,
    fourcc,
    fps,
    frame_size_flat,
    isColor=True,
)

#   Initializes both human segmentation and ball and rim detection models
yolo = myYOLO()

#   Initializes SIFT and FLANN algorithms
sift = cv2.SIFT_create(
    nfeatures=0, nOctaveLayers=5, contrastThreshold=0.07, edgeThreshold=50, sigma=1.6
)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict(checks=100))
reader = easyocr.Reader(["en"])

#   Gets court and scoreboard keypoints
court_keypoints = get_keypoints(court_kp_file_path, frame)
scoreboard_keypoints = get_keypoints(sb_kp_file_path, frame, "scoreboard")

#   Calculates all other court keypoints
court_keypoints = find_other_court_points(court_keypoints)
court_polygon = get_court_poly(court_keypoints, frame.shape)

#   Gets human bounding boxes inside the court
prev_bboxes, curr_bboxes_conf, prev_polys = yolo.detect_persons(frame)
filtered_indices = [
    index
    for index, bbox in enumerate(prev_bboxes)
    if bbox_in_polygon(round_bbox(bbox), court_polygon)
]
prev_bboxes = [prev_bboxes[index] for index in filtered_indices]
curr_bboxes_conf = [curr_bboxes_conf[index] for index in filtered_indices]
prev_polys = [prev_polys[index] for index in filtered_indices]
prev_bboxes, prev_polys, curr_bboxes_conf = filter_bboxes(
    prev_bboxes, prev_polys, curr_bboxes_conf
)

#   Create a class object for each person on court
KMEANS = generate_cluster(prev_polys, frame)
players_in_prev_frame = []
for bbox, poly, conf in zip(prev_bboxes, prev_polys, curr_bboxes_conf):
    bbox = round_bbox(bbox)
    if bbox_in_polygon(bbox, court_polygon):
        team_id = get_team(poly, frame.copy(), KMEANS)
        new_player = Player(1, bbox, poly, team_id)
        players_in_prev_frame.append(new_player)

#   Time counting
time_yolo1 = 0.0
time_yolo2 = 0.0
time_frame_h = 0.0
time_track = 0.0
time_draw = 0.0
time_end = 0.0

#   Start tracking
prev_frame = frame.copy()
prev_frame_court = frame.copy()
prev_bboxes_court = prev_bboxes.copy()
lost_players = []
player_data = []
confirmed_numbers = {"0": [], "1": []}
while True:
    ret, curr_frame = cap.read()
    if not ret:
        break

    #   Get detections in every frame
    if FRAME_COUNTER % 1 == 0:
        curr_frame_clean = curr_frame.copy()

        before1 = time.time()
        #   Gets human bounding boxes in current frame
        curr_bboxes, curr_bboxes_conf, curr_polys = yolo.detect_persons(curr_frame)
        curr_bboxes, curr_polys, curr_bboxes_conf = filter_bboxes(
            curr_bboxes, curr_polys, curr_bboxes_conf
        )
        time_yolo1 += time.time() - before1

        before = time.time()
        #   Gets ball and rim detections
        ball_rim_bboxes, ball_rim_conf, ball_rim_classes = (
            yolo.detect_basketball_objects(curr_frame)
        )
        time_yolo2 += time.time() - before

    #   Tracks court every 5th frame (higher accuracy)
    if FRAME_COUNTER % 5 == 0:
        before = time.time()
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
        time_frame_h += time.time() - before

        #   Extract previous frame points
        prev_frame_points = [
            value for value in court_keypoints.values() if value is not None
        ]
        prev_frame_keys = [
            key for key, value in court_keypoints.items() if value is not None
        ]

        #   Calculate current frame court keypoints using the homography matrix
        curr_frame_points = cv2.perspectiveTransform(
            np.array(prev_frame_points, dtype=np.float32).reshape(-1, 1, 2), H
        ).reshape(-1, 2)

        #   Sets new points
        for key in court_keypoints:
            court_keypoints[key] = None
        for i, key in enumerate(prev_frame_keys):
            # curr_frame = draw_court_point(curr_frame, curr_frame_points[i], key)

            court_keypoints[key] = (
                round(curr_frame_points[i][0]),
                round(curr_frame_points[i][1]),
            )

        #   Creates copies for next iteration
        prev_bboxes_court = curr_bboxes.copy()
        prev_frame_court = curr_frame_clean.copy()

    #   Tracks players in every frame
    if FRAME_COUNTER % 1 == 0:
        court_polygon = get_court_poly(court_keypoints, curr_frame.shape)
        # cv2.polylines(
        #     curr_frame,
        #     np.int32([court_polygon]),
        #     isClosed=True,
        #     color=(0, 0, 255),
        #     thickness=1,
        # )

        before = time.time()
        players_in_frame, lost_players, player_data, confirmed_numbers = track_players(
            players_in_prev_frame,
            lost_players,
            FRAME_COUNTER,
            curr_bboxes,
            curr_polys,
            curr_bboxes_conf,
            KMEANS,
            curr_frame_clean,
            court_polygon,
            player_data,
            reader,
            confirmed_numbers,
        )
        time_track += time.time() - before

        #   Find the game ball
        players_in_frame, game_ball = find_game_ball(
            ball_rim_bboxes,
            ball_rim_conf,
            ball_rim_classes,
            court_polygon,
            players_in_frame,
        )

        before = time.time()
        #   Draw
        points_in_frame = dict(
            (key, coord)
            for key, coord in court_keypoints.items()
            if is_point_in_frame(coord, curr_frame.shape[1], curr_frame.shape[0])
        )
        annotated_frame, annotated_flat, players_in_frame = draw_images(
            points_in_frame, players_in_frame, curr_frame.copy(), flat_court.copy()
        )

        #   Writes frame to video
        out_flat.write(annotated_flat)
        out_original.write(annotated_frame)
        time_draw += time.time() - before

        #   Early stoppage for debugging
        # if FRAME_COUNTER == 300:
        #     break

        prev_bboxes = curr_bboxes
        prev_polys = curr_polys
        prev_frame = curr_frame_clean
        players_in_prev_frame = players_in_frame

    print(f"FRAMES: {FRAME_COUNTER}/{TOTAL_FRAMES}")
    FRAME_COUNTER += 1
    time_end += time.time() - before1

#   Closes video and windows
cap.release()
cv2.destroyAllWindows()
out_original.release()
out_flat.release()

create_result_json(
    player_data, players_in_frame, lost_players, FRAME_COUNTER, result_data_file_path
)

#   Output time measurements
print(f"YOLO seg: {time_yolo1/FRAME_COUNTER}")
print(f"YOLO ball: {time_yolo2/FRAME_COUNTER}")
print(f"Frame transform H: {time_frame_h/FRAME_COUNTER*5}")
print(f"Tracking: {time_track/FRAME_COUNTER}")
print(f"Draw normal: {time_draw/FRAME_COUNTER}")
print(f"Total: {time_end/FRAME_COUNTER}")
