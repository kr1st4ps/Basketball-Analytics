import time
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch
from detectron2.structures import Instances
from shapely.geometry import Polygon, Point, box
from ultralytics import YOLO



class myYOLO:
    def __init__(self):
        # self.model = YOLO("yolov8m-seg.pt")
        self.model = YOLO("yolov8m.pt")

    def get_bboxes(self, img):
        results = self.model.predict(img, classes=0)
        
        return results[0].boxes.xyxy
    

def filter_bboxes_by_polygon(bboxes, polygon):
    """
    Filter bounding boxes that intersect with a given polygon.

    Args:
        bboxes (torch.Tensor): Tensor of shape (N, 4) representing bounding boxes in xyxy format.
        polygon (list): List of tuples representing the vertices of the polygon.

    Returns:
        torch.Tensor: Tensor containing filtered bounding boxes.
    """
    filtered_bboxes = []
    for bbox in bboxes:
        bbox_vertices = [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]
        if any(point_in_polygon(vertex, polygon) for vertex in bbox_vertices):
            filtered_bboxes.append(bbox)
    return filtered_bboxes

def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using ray casting algorithm.

    Args:
        point (tuple): Tuple representing the (x, y) coordinates of the point.
        polygon (list): List of tuples representing the vertices of the polygon.

    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def draw_bboxes_on_image(image, bboxes, color=(255, 0, 0), thickness=2):
    """
    Draw bounding boxes on an image.

    Args:
        image (numpy.ndarray): The input image.
        bboxes (torch.Tensor): Tensor of shape (N, 4) representing bounding boxes in xyxy format.
        color (tuple, optional): Color of the bounding box lines in BGR format. Defaults to (0, 255, 0).
        thickness (int, optional): Thickness of the bounding box lines. Defaults to 2.

    Returns:
        numpy.ndarray: The image with bounding boxes drawn on it.
    """
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox.tolist()
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return image