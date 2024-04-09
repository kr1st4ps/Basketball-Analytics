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
        # self.model = YOLO("yolov8m-seg.pt")
        self.model = YOLO("yolov8m.pt")

    def get_bboxes(self, img):
        """
        Gets only bboxes of class 0 (person).
        """
        results = self.model.predict(img, classes=0)

        return results[0].boxes.xyxy


def draw_bboxes(frame, bboxes, court_polygon):
    """
    Draws YOLO bboxes on frame if they are within a polygon.
    """
    for bbox in bboxes:
        x1, y1, x2, y2 = [round(float(coord)) for coord in bbox]
        if (
            cv2.pointPolygonTest(court_polygon, ((x1 + ((x2 - x1) / 2)), y2), False)
            >= 0
        ):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return frame
