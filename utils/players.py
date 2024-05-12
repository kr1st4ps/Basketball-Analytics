"""
Player utils.
"""

import re
import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
import pytesseract


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

        self.team = team
        self.incorrect_team = 0

        self.number = ""
        self.num_conf = 0

        self.has_ball = False

        self.start_frame = start_frame
        self.last_seen = start_frame
        self.num_assign_frame = -1
        self.end_frame = None

    def __str__(self):
        return f"{self.id}.\tFrames[{self.start_frame}-{self.end_frame}]\tHistory{self.poly_history}"

    def __del__(self):
        None

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
        if len(self.bbox_history) > 5:
            self.bbox_history.pop()
            self.poly_history.pop()

    def lost(self, frame_id):
        """
        Handles player tracking being lost.
        """
        self.end_frame = frame_id
        #   TODO save to file

    def check_team(self, frame, kmeans):
        c = get_team(self.bbox_history[0], self.poly_history[0], frame)
        label = get_label(kmeans, c)
        if label != self.team:
            self.incorrect_team += 1
            if self.incorrect_team == 3:
                self.team = label
                self.incorrect_team = 0
        else:
            self.incorrect_team = 0


def bb_intersection_over_union(bbox_a, bbox_b):
    """
    Finds IoU (intersection over union) value of two bboxes.
    """
    # x_max = max(bbox_a[0], bbox_b[0])
    # y_max = max(bbox_a[1], bbox_b[1])
    # x_min = min(bbox_a[2], bbox_b[2])
    # y_min = min(bbox_a[3], bbox_b[3])

    # intersection_area = max(0, x_min - x_max + 1) * max(0, y_min - y_max + 1)
    intersection_width = max(
        0, min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0]) + 1
    )
    intersection_height = max(
        0, min(bbox_a[3], bbox_b[3]) - max(bbox_a[1], bbox_b[1]) + 1
    )
    intersection_area = intersection_width * intersection_height

    bbox_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    bbox_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    iou = intersection_area / float(bbox_a_area + bbox_b_area - intersection_area)

    return iou


def poly_intersection_over_union(poly_a, poly_b):
    poly_a = Polygon(poly_a)
    poly_b = Polygon(poly_b)
    i = poly_a.intersection(poly_b).area
    u = poly_a.union(poly_b).area
    iou = i / u

    return iou


def filter_bboxes(bboxes, polys, confidences, iou_threshold=0.5):
    """
    Filters out bounding boxes that overlap significantly with others.
    """
    filtered_bboxes = []
    filtered_polys = []
    filtered_confidences = []

    keep = [True] * len(bboxes)

    for i in range(len(bboxes)):
        if not keep[i]:
            continue

        for j in range(i + 1, len(bboxes)):
            if not keep[j]:
                continue

            iou = bb_intersection_over_union(bboxes[i], bboxes[j])
            if iou > iou_threshold:
                if confidences[i] >= confidences[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    for i in range(len(bboxes)):
        if keep[i]:
            filtered_bboxes.append(bboxes[i])
            filtered_polys.append(polys[i])
            filtered_confidences.append(confidences[i])

    return filtered_bboxes, filtered_polys, filtered_confidences


def get_team(bbox, poly, img):
    c = (0, 0, 0)
    poly_mask = np.zeros_like(img[:, :, 0])
    poly = np.array([poly], dtype=np.int32)
    cv2.fillPoly(poly_mask, poly, color=255)

    bbox_mask = np.zeros_like(img[:, :, 0])
    cv2.rectangle(bbox_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, thickness=-1)

    combined_mask = cv2.bitwise_and(poly_mask, bbox_mask)

    nonzero_coords = np.column_stack(np.where(combined_mask != 0))

    b, g, r = cv2.split(img)

    bins = 5
    hist_b = cv2.calcHist([b], [0], poly_mask, [bins], [0, 256])
    hist_g = cv2.calcHist([g], [0], poly_mask, [bins], [0, 256])
    hist_r = cv2.calcHist([r], [0], poly_mask, [bins], [0, 256])
    hist_concatenated = np.concatenate(
        [hist_b.flatten(), hist_g.flatten(), hist_r.flatten()]
    )

    for coord in nonzero_coords:
        x, y = coord
        c += img[x][y]

    # return (c / len(nonzero_coords)).round()
    return hist_concatenated


def get_label(kmeans, value):
    # color_lab = np.array(value).reshape(1, 1, 3).astype(np.uint8)
    # colors_lab = cv2.cvtColor(colors_lab, cv2.COLOR_RGB2Lab)
    # label = kmeans.predict(color_lab.reshape(-1, 3))[0]
    label = kmeans.predict(value.reshape(1, -1))[0]

    return label


def get_team_coef(bboxes, polys, img):
    all_colors = []
    all_hists = []
    for bbox, poly in zip(bboxes, polys):
        c = (0, 0, 0)
        bbox = [round(coord) for coord in bbox.cpu().numpy().tolist()]

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

        bbox_mask = np.zeros_like(img[:, :, 0])
        cv2.rectangle(
            bbox_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, thickness=-1
        )

        combined_mask = cv2.bitwise_and(poly_mask, bbox_mask)

        nonzero_coords = np.column_stack(np.where(combined_mask != 0))

        for coord in nonzero_coords:
            x, y = coord
            c += img[x][y]

        all_colors.append((c / len(nonzero_coords)).round())
        all_hists.append(hist_concatenated)

    concatenated_hist_values = np.vstack(all_hists)

    # colors = np.array(all_colors)
    # colors_lab = np.apply_along_axis(
    #     lambda x: np.reshape(np.array([x]), (1, 1, 3)).astype(np.uint8), 1, colors
    # )
    # colors_lab = np.array(all_colors).reshape(-1, 1, 3).astype(np.uint8)
    # colors_lab = cv2.cvtColor(colors_lab, cv2.COLOR_RGB2Lab)

    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters, random_state=3).fit(
        # colors_lab.reshape(-1, 3)
        concatenated_hist_values
    )

    return kmeans


def read_number(bbox, img):
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1
    y1 += 0.15 * h
    y2 -= 0.6 * h
    x1 += 0.15 * w
    x2 -= 0.15 * w
    y1 = round(y1)
    y2 = round(y2)
    x1 = round(x1)
    x2 = round(x2)
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
