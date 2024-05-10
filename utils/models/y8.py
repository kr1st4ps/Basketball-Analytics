"""
Actions with YOLO models.
"""

import cv2
from ultralytics import YOLO


class myYOLO:
    """
    YOLO model class.
    """

    def __init__(self):
        self.model_person = YOLO("yolov8x-seg.pt")
        self.model_bball = YOLO("ball-rim.pt")

    def detect_persons(self, img):
        """
        Gets only bboxes of class 0 (person).
        """
        results = self.model_person.predict(img, classes=0)

        return results[0].boxes.xyxy, results[0].boxes.conf, results[0].masks.xy

    def detect_basketball_objects(self, img):
        results = self.model_bball.predict(img, conf=0.5)

        return results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls


def draw_bboxes(frame, bboxes, court_polygon):
    """
    Draws YOLO bboxes on frame if they are within a polygon.
    """
    for bbox in bboxes:
        bbox_rounded = [round(float(coord)) for coord in bbox]
        if bbox_in_polygon(bbox_rounded, court_polygon):
            x1, y1, x2, y2 = bbox_rounded
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return frame


def bbox_in_polygon(bbox, polygon):
    """
    Checks if bbox is inside a polygon.
    """
    if (
        cv2.pointPolygonTest(
            polygon, ((bbox[0] + ((bbox[2] - bbox[0]) / 2)), bbox[3]), True
        )
        > -10
    ):
        return True
    else:
        return False
