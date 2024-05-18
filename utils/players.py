"""
Player utils.
"""

import json
import re
import cv2
import numpy as np
from sklearn.cluster import KMeans
import pytesseract

from utils.functions import bb_intersection_over_union, round_bbox
from utils.models.y8 import bbox_in_polygon


class Player:
    """
    Player class, containing all information of a detected and tracked player.
    """

    _id_counter = 0

    def __init__(self, start_frame, bbox, poly, team):
        type(self)._id_counter += 1
        self.id = self._id_counter

        self.bbox_history = [bbox]
        self.poly_history = [poly]
        self.total_dist = 0

        self.team = team
        self.incorrect_team = 0

        self.number = ""
        self.num_conf = 0

        self.has_ball = False
        self.has_ball_frames = 0

        self.start_frame = start_frame
        self.last_seen = start_frame
        self.num_assign_frame = -1
        self.end_frame = None

    def __str__(self):
        return f"{self.id}.\tFrames[{self.start_frame}-{self.end_frame}]\tHistory{self.poly_history}"

    def __eq__(self, other):
        return self.id == other.id

    def update(self, bbox, poly, frame_id, num, num_conf):
        """
        Updates player info.
        """
        self.bbox_history.insert(0, bbox)
        self.poly_history.insert(0, poly)
        self.last_seen = frame_id
        if num_conf > self.num_conf and len(str(num)) > 0:
            self.number = num
            self.num_conf = num_conf
            self.num_assign_frame = frame_id
        if len(self.poly_history) > 5:
            self.poly_history.pop()

    def update_distance(self, dist):
        """
        Updates total distance detection has traveled
        """
        self.total_dist += dist


def get_team(poly, img, kmeans):
    """
    Returns team ID based on player histogram and previously defined k-means cluster.
    """
    poly_mask = np.zeros_like(img[:, :, 0])
    poly = np.array([poly], dtype=np.int32)
    cv2.fillPoly(poly_mask, poly, color=255)

    b, g, r = cv2.split(img)

    bins = 5
    hist_b = cv2.calcHist([b], [0], poly_mask, [bins], [0, 256])
    hist_g = cv2.calcHist([g], [0], poly_mask, [bins], [0, 256])
    hist_r = cv2.calcHist([r], [0], poly_mask, [bins], [0, 256])
    hist_concatenated = np.concatenate(
        [hist_b.flatten(), hist_g.flatten(), hist_r.flatten()]
    )

    team_id = kmeans.predict(hist_concatenated.reshape(1, -1))[0]

    return team_id


def generate_cluster(polys, img):
    """
    Generates a k-means cluster.
    """
    all_hists = []
    for poly in polys:
        poly_mask = np.zeros_like(img[:, :, 0])
        poly = np.array([poly], dtype=np.int32)
        cv2.fillPoly(poly_mask, poly, color=255)

        b, g, r = cv2.split(img)

        bins = 5
        hist_b = cv2.calcHist([b], [0], poly_mask, [bins], [0, 256])
        hist_g = cv2.calcHist([g], [0], poly_mask, [bins], [0, 256])
        hist_r = cv2.calcHist([r], [0], poly_mask, [bins], [0, 256])
        hist_concatenated = np.concatenate(
            [hist_b.flatten(), hist_g.flatten(), hist_r.flatten()]
        )

        all_hists.append(hist_concatenated)

    concatenated_hist_values = np.vstack(all_hists)

    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters, random_state=3).fit(
        concatenated_hist_values
    )

    return kmeans


def read_number(bbox, img):
    """
    Reads number from player bbox (if possible).
    """
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    width = x2 - x1
    y1 = round(y1 + 0.15 * height)
    y2 = round(y2 - 0.6 * height)
    x1 = round(x1 + 0.15 * width)
    x2 = round(x2 - 0.15 * width)

    roi = img[y1:y2, x1:x2]
    new_width = int(roi.shape[1] * 4)
    new_height = int(roi.shape[0] * 4)
    roi = cv2.resize(roi, (new_width, new_height))

    thresh_img = cv2.threshold(
        cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]

    num = pytesseract.image_to_data(
        thresh_img, config="--psm 6", output_type=pytesseract.Output.DICT
    )

    best_conf = 0
    best_text = ""
    for text, conf in zip(num["text"], num["conf"]):
        if conf >= best_conf:
            best_conf = conf
            best_text = re.sub(r"\D", "", text)

    if best_conf < 70:
        best_text = ""
        best_conf = 0

    try:
        first_two_digits = best_text[:2]
        best_text = int(first_two_digits)
    except:
        best_text = ""
        best_conf = 0

    return best_text, best_conf


def to_json(player):
    """
    Transforms Player object into a JSON.
    """
    if player.has_ball_frames == 0:
        ball_percentage = 0
    else:
        ball_percentage = player.has_ball_frames / (
            player.end_frame - player.start_frame
        )
    json_obj = {
        "id": [int(player.id)],
        "team": int(player.team),
        "number": str(player.number),
        "number_conf": int(player.num_conf),
        "appeared": int(player.start_frame),
        "disappeared": int(player.end_frame),
        "distance": float(round(player.total_dist / 100, 3)),
        "frames_ball": int(player.has_ball_frames),
        "%_ball": float(ball_percentage),
    }
    return json_obj


def track_players(
    players_in_prev_frame,
    lost_players,
    frame_cnt,
    curr_bboxes,
    curr_polys,
    curr_conf,
    kmeans,
    frame,
    court_polygon,
    player_data,
    iou_thresh=0.0,
):
    """
    Tracks players by comparing bboxes or distance with the previous frame.
    """

    #   Finds all bbox intersections
    found_intersections = []
    players_to_check = [
        player for player in lost_players if frame_cnt - player.last_seen < 5
    ] + players_in_prev_frame
    for bbox, poly, _ in zip(curr_bboxes, curr_polys, curr_conf):
        bbox = round_bbox(bbox)
        for player in players_to_check:
            iou = bb_intersection_over_union(bbox, player.bbox_history[0])
            if iou > iou_thresh:
                found_intersections.append((player, bbox, poly, iou))

    #   Sorts intersections by IoU value
    found_intersections.sort(key=lambda x: x[3], reverse=True)

    #   Checks which players have multiple intersections
    intersection_count = {}
    for item in found_intersections:
        key = item[0].id
        intersection_count[key] = intersection_count.get(key, 0) + 1

    #   Goes through found intersections, reads their number and updates players
    found_bboxes = []
    found_players = []
    players_in_frame = []
    for intersection in found_intersections:
        player, bbox, poly, iou = intersection
        if player.id not in found_players and bbox not in found_bboxes:
            if intersection_count[player.id] > 1 and iou < 0.5:
                team_id = get_team(poly, frame.copy(), kmeans)
                if team_id != player.team:
                    intersection_count[player.id] -= 1
                    continue

            num, num_conf = read_number(bbox, frame.copy())
            player.update(bbox, poly, frame_cnt, num, num_conf)

            players_in_frame.append(player)

            found_bboxes.append(bbox)
            found_players.append(player.id)
        intersection_count[player.id] -= 1

    #   Updates lost players
    lost_players = [
        player for player in lost_players if player.id not in found_players
    ] + [player for player in players_in_prev_frame if player.id not in found_players]

    #   Finds all remaining players and detections gets distance between them
    found_pairs = []
    for bbox, poly, _ in zip(curr_bboxes, curr_polys, curr_conf):
        bbox = round_bbox(bbox)
        if bbox not in found_bboxes and bbox_in_polygon(bbox, court_polygon):
            for player in lost_players:
                player_center = np.mean(player.bbox_history[0], axis=0)
                bbox_center = np.mean(bbox, axis=0)
                dist = np.linalg.norm(player_center - bbox_center)

                found_pairs.append((player, bbox, poly, dist))

    #   Sorts pairs by shortest distance
    found_pairs.sort(key=lambda x: x[3])

    #   Checks if pair is valid and updates those players
    for pair in found_pairs:
        player, bbox, poly, dist = pair
        team_id = get_team(poly, frame.copy(), kmeans)
        if (
            frame_cnt - player.last_seen >= 5
            and dist < 150
            and player.team == team_id
            and player.id not in found_players
            and bbox not in found_bboxes
        ):
            num, num_conf = read_number(bbox, frame.copy())
            player.update(bbox, poly, frame_cnt, num, num_conf)

            players_in_frame.append(player)

            found_bboxes.append(bbox)
            found_players.append(player.id)

    #   Updates lost players
    lost_players = [player for player in lost_players if player.id not in found_players]

    #   Finds any remaining detections (new players)
    new_players = [
        (bbox, poly, conf)
        for bbox, poly, conf in zip(curr_bboxes, curr_polys, curr_conf)
        if round_bbox(bbox) not in found_bboxes
    ]

    #   Finds any players with duplicate team and number
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

    #   Deletes duplicates
    for player in players_in_frame:
        if player.id in ids_to_remove:
            player.end_frame = frame_cnt
            append_player(player, player_data)
    players_in_frame = [
        player for player in players_in_frame if player.id not in ids_to_remove
    ]

    #   Creates new players
    for bbox, poly, _ in new_players:
        try:
            bbox = round_bbox(bbox)
        except:
            bbox = bbox
        if bbox_in_polygon(bbox, court_polygon):
            team_id = get_team(poly, frame.copy(), kmeans)
            new_player = Player(frame_cnt, bbox, poly, team_id)
            players_in_frame.append(new_player)

    return players_in_frame, lost_players, player_data


def create_result_json(
    player_data, players_in_frame, lost_players, frame_cnt, output_path
):
    """
    Generates the end result JSON with all collected data.
    """
    #   Collects data from players that were still on the court
    for player in players_in_frame:
        player.end_frame = frame_cnt
        append_player(player, player_data)

    #   Collects data from players that were last at last frame
    for player in lost_players:
        player.end_frame = player.last_seen
        append_player(player, player_data)

    #   Collects team specific data
    team0_total_detections = 0
    team1_total_detections = 0
    team0_total_distance = 0
    team1_total_distance = 0
    team0_total_ball_frames = 0
    team1_total_ball_frames = 0
    for player in player_data:
        if player["team"] == 0:
            team0_total_detections += 1
            team0_total_distance += player["distance"]
            team0_total_ball_frames += player["frames_ball"]
        if player["team"] == 1:
            team1_total_detections += 1
            team1_total_distance += player["distance"]
            team1_total_ball_frames += player["frames_ball"]
    team0_possesion = round(team0_total_ball_frames / frame_cnt, 2)
    team1_possesion = round(team1_total_ball_frames / frame_cnt, 2)

    #   Creates JSON objects
    team0_data = {
        "total_detections": team0_total_detections,
        "total_distance": round(team0_total_distance, 3),
        "avg_distance": round(team0_total_distance / team0_total_detections, 3),
        "possesion": team0_possesion,
    }
    team1_data = {
        "total_detections": team1_total_detections,
        "total_distance": round(team1_total_distance, 3),
        "avg_distance": round(team1_total_distance / team1_total_detections, 3),
        "possesion": team1_possesion,
    }
    total_data = {
        "total_detections": team0_total_detections + team1_total_detections,
        "total_distance": round(team0_total_distance + team1_total_distance, 3),
        "avg_distance": round(
            (team0_total_distance + team1_total_distance)
            / (team0_total_detections + team1_total_detections),
            3,
        ),
        "possesion": {
            "team0": team0_possesion,
            "dead_ball": 1.0 - team0_possesion - team1_possesion,
            "team1": team1_possesion,
        },
    }

    end_json = {
        "players": player_data,
        "teams": {
            "team0": team0_data,
            "team1": team1_data,
        },
        "total": total_data,
    }

    #   Writes file
    with open(output_path, "w") as file:
        file.write(json.dumps(end_json, indent=4))


def append_player(player, player_data):
    """
    Appends player data.
    """
    if player.number != "":
        for other_player in player_data:
            if other_player["number"] == str(player.number) and other_player[
                "team"
            ] == int(player.team):
                other_player["id"].append(int(player.id))
                other_player["number_conf"] = (
                    player.num_conf
                    if player.num_conf > other_player["number_conf"]
                    else other_player["number_conf"]
                )
                other_player["disappeared"] = player.end_frame
                other_player["distance"] += float(round(player.total_dist / 100, 3))
                other_player["frames_ball"] += int(player.has_ball_frames)
                if other_player["frames_ball"] > 0:
                    other_player["%_ball"] = other_player["frames_ball"] / (
                        other_player["disappeared"] - other_player["appeared"]
                    )

                return

    player_data.append(to_json(player))
