"""
Actions with YOLO models.
"""

import cv2
from ultralytics import YOLO

from utils.functions import bb_intersection_over_union


class myYOLO:
    """
    YOLO model class.
    """

    def __init__(self):
        self.model_person = YOLO("yolov8x-seg.pt")
        self.model_bball = YOLO("ball-rim.pt")

    def detect_persons(self, img):
        """
        Gets class 0 (person) detections.
        """
        results = self.model_person.predict(img, classes=0, device="mps")

        return results[0].boxes.xyxy, results[0].boxes.conf, results[0].masks.xy

    def detect_basketball_objects(self, img):
        """
        Gets ball and rim detections.
        """
        results = self.model_bball.predict(img, conf=0.5, device="mps")

        return results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls


def bbox_in_polygon(bbox, polygon):
    """
    Checks if bbox is inside a polygon.
    """
    if (
        cv2.pointPolygonTest(polygon, (bbox[2], bbox[3]), False) >= 0
        or cv2.pointPolygonTest(polygon, (bbox[0], bbox[3]), False) >= 0
    ):
        return True
    else:
        return False


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
