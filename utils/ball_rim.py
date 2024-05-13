"""
Ball and rim helper functions.
"""

import cv2
import numpy as np


from utils.functions import bbox_intersect


def find_game_ball(
    ball_rim_bboxes, ball_rim_conf, ball_rim_classes, court_polygon, players_in_frame
):
    """
    Finds which ball is the real one (if multiple) and finds which player has it (if any).
    """

    #   Checks if any detection was found
    if len(ball_rim_classes) > 0:

        #   Finds how many balls were detected and takes the one that is the closest to the court
        ball_rim_classes = [int(cls) for cls in ball_rim_classes]
        ball_count = ball_rim_classes.count(0)
        if ball_count > 1:
            balls = []
            for bbox, conf, cls in zip(
                ball_rim_bboxes, ball_rim_conf, ball_rim_classes
            ):
                if cls == 0:
                    ball_bbox = bbox.cpu().numpy()
                    center_x = (ball_bbox[0] + ball_bbox[2]) / 2
                    center_y = (ball_bbox[1] + ball_bbox[3]) / 2
                    dist = cv2.pointPolygonTest(
                        court_polygon, (center_x, center_y), True
                    )
                    balls.append((bbox, conf, cls, dist))
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

        #   Checks which players bboxes intersect with balls bbox
        ball_intersection_ids = []
        previous_owner = -1
        if game_ball:
            for player in players_in_frame:
                if player.has_ball:
                    previous_owner = player.id
                player.has_ball = False
                if bbox_intersect(player.bbox_history[0], game_ball[0][0]):
                    player_center = np.mean(player.bbox_history[0], axis=0)
                    ball_bbox = game_ball[0][0].cpu().numpy()
                    ball_center = (
                        (ball_bbox[0] + ball_bbox[2]) / 2,
                        (ball_bbox[1] + ball_bbox[3]) / 2,
                    )
                    dist = np.linalg.norm(player_center - ball_center)
                    ball_intersection_ids.append((player.id, dist))

            if len(ball_intersection_ids) >= 1:
                if len(ball_intersection_ids) > 1:
                    ball_intersection_ids = sorted(
                        ball_intersection_ids, key=lambda x: x[1]
                    )

                for player in players_in_frame:
                    if player.id == ball_intersection_ids[0][0]:
                        player.has_ball = True
                        break
            else:
                for player in players_in_frame:
                    if player.id == previous_owner:
                        player.has_ball = True
                        break

    for player in players_in_frame:
        if player.has_ball:
            player.has_ball_frames += 1
            break

    return players_in_frame
