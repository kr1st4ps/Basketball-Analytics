"""
Main.
"""

import os
import sys
import time
import cv2
import numpy as np


from utils.court import draw_court_point, get_court_poly, get_keypoints
from utils.functions import (
    bbox_intersect,
    draw_images,
    find_frame_transform,
    find_other_court_points,
    find_view_homography,
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
    read_number,
)

start_time = time.time()

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

first_iteration = time.time()
time_yolo1 = 0.0
time_yolo2 = 0.0
time_frame_h = 0.0
time_track_get_iou = 0.0
time_track_check_iou = 0.0
time_track_get_dist = 0.0
time_track_check_dist = 0.0
time_track_remove_add = 0.0
time_track_find_ball = 0.0
time_draw = 0.0
time_end = 0.0

#   Start tracking
prev_frame = frame.copy()
prev_frame_court = frame.copy()
prev_bboxes_court = prev_bboxes.copy()
frame_counter = 1
IOU_THRESHOLD = 0.0
lost_players = []
orange = (0, 165, 255)
team0_confirmed_numbers = []
team1_confirmed_numbers = []
while True:
    ret, curr_frame = cap.read()
    if not ret:
        break
    if frame_counter % 1 == 0:
        curr_frame_copy = curr_frame.copy()

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

    if frame_counter % 5 == 0:
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
        # cv2.polylines(
        #     curr_frame,
        #     np.int32([court_polygon]),
        #     isClosed=True,
        #     color=(255, 0, 0),
        #     thickness=1,
        # )

        before = time.time()
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
        time_track_get_iou += time.time() - before

        found_intersections.sort(key=lambda x: x[3], reverse=True)
        intersection_count = {}
        for item in found_intersections:
            key = item[0].id
            intersection_count[key] = intersection_count.get(key, 0) + 1

        before = time.time()
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

                num, num_conf = read_number(bbox, curr_frame.copy())
                player.update(bbox, poly, frame_counter, num, num_conf)
                if player.team == 0:
                    if (
                        player.num_conf > 80
                        and player.number not in team0_confirmed_numbers
                    ):
                        team0_confirmed_numbers.append(player.number)
                else:
                    if (
                        player.num_conf > 80
                        and player.number not in team1_confirmed_numbers
                    ):
                        team1_confirmed_numbers.append(player.number)

                players_in_frame.append(player)

                found_bboxes.append(bbox)
                found_players.append(player.id)
            intersection_count[player.id] -= 1
        time_track_check_iou += time.time() - before

        lost_players = [
            player for player in lost_players if player.id not in found_players
        ] + [
            player for player in players_in_prev_frame if player.id not in found_players
        ]

        before = time.time()
        found_pairs = []
        for bbox, poly, conf in zip(curr_bboxes, curr_polys, curr_bboxes_conf):
            bbox = [round(coord) for coord in bbox.cpu().numpy().tolist()]
            if bbox not in found_bboxes and bbox_in_polygon(bbox, court_polygon):
                for player in lost_players:
                    player_center = np.mean(player.bbox_history[0], axis=0)
                    bbox_center = np.mean(bbox, axis=0)
                    dist = np.linalg.norm(player_center - bbox_center)

                    found_pairs.append((player, bbox, poly, dist))
        time_track_get_dist += time.time() - before

        before = time.time()
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
                num, num_conf = read_number(bbox, curr_frame.copy())
                player.update(bbox, poly, frame_counter, num, num_conf)
                if player.team == 0:
                    if (
                        player.num_conf > 80
                        and player.number not in team0_confirmed_numbers
                    ):
                        team0_confirmed_numbers.append(player.number)
                else:
                    if (
                        player.num_conf > 80
                        and player.number not in team1_confirmed_numbers
                    ):
                        team1_confirmed_numbers.append(player.number)

                players_in_frame.append(player)

                found_bboxes.append(bbox)
                found_players.append(player.id)
        time_track_check_dist += time.time() - before

        before = time.time()
        lost_players = [
            player for player in lost_players if player.id not in found_players
        ]
        new_players = [
            (bbox, poly, conf)
            for bbox, poly, conf in zip(curr_bboxes, curr_polys, curr_bboxes_conf)
            if [round(coord) for coord in bbox.cpu().numpy().tolist()]
            not in found_bboxes
        ]

        players_in_frame = sorted(
            players_in_frame, key=lambda x: x.num_assign_frame, reverse=True
        )
        original_players = []
        ids_to_remove = []
        for player in players_in_frame:
            if (player.team, player.number) in original_players:
                ids_to_remove.append(player.id)
                new_players.append((player.bbox_history[0], player.poly_history[0], 0))
            elif player.number != "":
                original_players.append((player.team, player.number))

        players_in_frame = [
            player for player in players_in_frame if player.id not in ids_to_remove
        ]

        for bbox, poly, conf in new_players:
            try:
                bbox = [round(coord) for coord in bbox.cpu().numpy().tolist()]
            except:
                bbox = bbox
            if bbox_in_polygon(bbox, court_polygon):
                c = get_team(bbox, poly, curr_frame_copy.copy())
                label = get_label(KMEANS, c)
                new_player = Player(frame_counter, bbox, poly, label)
                players_in_frame.append(new_player)
        time_track_remove_add += time.time() - before

        # for player in players_in_frame:
        #     if (frame_counter - player.start_frame) % 30 == 0:
        #         player.check_team(curr_frame, KMEANS)

        #   Draw bboxes
        before = time.time()
        #   Find the game ball
        if len(ball_rim_classes) > 0:
            ball_rim_classes = [int(cls) for cls in ball_rim_classes]
            rims = [
                (bbox, conf, cls)
                for bbox, conf, cls in zip(
                    ball_rim_bboxes, ball_rim_conf, ball_rim_classes
                )
                if cls == 1
            ]
            ball_count = ball_rim_classes.count(0)
            rim_count = len(ball_rim_classes) - ball_count
            if ball_count > 1:
                balls = [
                    (
                        bbox,
                        conf,
                        cls,
                        cv2.pointPolygonTest(
                            court_polygon,
                            (
                                (bbox.cpu().numpy()[0] + bbox.cpu().numpy()[2]) / 2,
                                (bbox.cpu().numpy()[1] + bbox.cpu().numpy()[3]) / 2,
                            ),
                            True,
                        ),
                    )
                    for bbox, conf, cls in zip(
                        ball_rim_bboxes, ball_rim_conf, ball_rim_classes
                    )
                    if cls == 0
                ]
                game_ball = sorted(balls, key=lambda x: x[3], reverse=True)
            elif ball_count == 1:
                game_ball = [
                    (bbox, conf, cls)
                    for bbox, conf, cls in zip(
                        ball_rim_bboxes, ball_rim_conf, ball_rim_classes
                    )
                    if cls == 0
                ]
            else:
                game_ball = None

            ball_owner_set = False
            ball_intersection_ids = []
            if game_ball:
                for idx, player in enumerate(players_in_frame):
                    player.has_ball = False
                    if bbox_intersect(player.bbox_history[0], game_ball[0][0]):
                        player_center = np.mean(player.bbox_history[0], axis=0)
                        ball_center = (
                            (
                                game_ball[0][0].cpu().numpy()[0]
                                + game_ball[0][0].cpu().numpy()[2]
                            )
                            / 2,
                            (
                                game_ball[0][0].cpu().numpy()[1]
                                + game_ball[0][0].cpu().numpy()[3]
                            )
                            / 2,
                        )
                        dist = np.linalg.norm(player_center - ball_center)
                        ball_intersection_ids.append((player.id, dist))

                if len(ball_intersection_ids) == 1:
                    for player in players_in_frame:
                        if player.id == ball_intersection_ids[0][0]:
                            player.has_ball = True
                elif len(ball_intersection_ids) > 1:
                    ball_intersection_ids = sorted(
                        ball_intersection_ids, key=lambda x: x[1]
                    )
                    for player in players_in_frame:
                        if player.id == ball_intersection_ids[0][0]:
                            player.has_ball = True
        time_track_find_ball += time.time() - before

        before = time.time()
        points_in_frame = dict(
            (key, coord)
            for key, coord in court_keypoints.items()
            if is_point_in_frame(coord, curr_frame.shape[1], curr_frame.shape[0])
        )
        #   Draw players on flat image
        annotated_frame, annotated_flat = draw_images(
            points_in_frame, players_in_frame, curr_frame.copy(), flat_court.copy()
        )
        out_flat.write(annotated_flat)
        cv2.putText(
            annotated_frame,
            f"FRAME: {frame_counter}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
        out_original.write(annotated_frame)
        time_draw += time.time() - before

        #   Writes frame to video

        # out.write(curr_frame)
        # if frame_counter == 100:
        #     break

        #   Sets current frame and bounding boxes as previous frames to be used for next frame
        prev_bboxes = curr_bboxes
        prev_polys = curr_polys
        prev_frame = curr_frame_copy
        players_in_prev_frame = players_in_frame

    # else:
    #     out.write(curr_frame)
    print(f"FRAMES: {frame_counter}/{TOTAL_FRAMES}")
    frame_counter += 1
    time_end += time.time() - before1

#   Closes video and windows
cap.release()
cv2.destroyAllWindows()
out_original.release()
out_flat.release()

print(f"YOLO seg: {time_yolo1/frame_counter}")
print(f"YOLO ball: {time_yolo2/frame_counter}")
print(f"Frame transform H: {time_frame_h/frame_counter*5}")
print(f"Get IoU: {time_track_get_iou/frame_counter}")
print(f"Check IoU: {time_track_check_iou/frame_counter}")
print(f"Get distance: {time_track_get_dist/frame_counter}")
print(f"Check distance: {time_track_check_dist/frame_counter}")
print(f"Remove/add players: {time_track_remove_add/frame_counter}")
print(f"Find ball: {time_track_find_ball/frame_counter}")
print(f"Draw normal: {time_draw/frame_counter}")
print(f"Total: {time_end/frame_counter}")
