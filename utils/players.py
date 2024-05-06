"""
Player utils.
"""

from shapely.geometry import Polygon


class Player:
    """
    Player class, containing all information of a detected and tracked player.
    """

    _id_counter = 0

    def __init__(self, start_frame, bbox, poly):
        type(self)._id_counter += 1
        self.id = self._id_counter

        self.bbox_history = [bbox]
        self.poly_history = [poly]

        self.start_frame = start_frame
        self.last_seen = start_frame
        self.end_frame = None

    def __str__(self):
        return f"{self.id}.\tFrames[{self.start_frame}-{self.end_frame}]\tHistory{self.poly_history}"

    def __del__(self):
        None

    def __eq__(self, other):
        return self.id == other.id

    def update(self, bbox, poly, frame_id):
        """
        Updates player info.
        """
        self.bbox_history.insert(0, bbox)
        self.poly_history.insert(0, poly)
        self.last_seen = frame_id
        if len(self.bbox_history) > 5:
            self.bbox_history.pop()
            self.poly_history.pop()

    def lost(self, frame_id):
        """
        Handles player tracking being lost.
        """
        self.end_frame = frame_id
        #   TODO save to file


def bb_intersection_over_union(bbox_a, bbox_b):
    """
    Finds IoU (intersection over union) value of two bboxes.
    """
    x_max = max(bbox_a[0], bbox_b[0])
    y_max = max(bbox_a[1], bbox_b[1])
    x_min = min(bbox_a[2], bbox_b[2])
    y_min = min(bbox_a[3], bbox_b[3])

    intersection_area = max(0, x_min - x_max + 1) * max(0, y_min - y_max + 1)

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


def filter_bboxes(bboxes, confidences, iou_threshold=0.5):
    """
    Filters out bounding boxes that overlap significantly with others.
    """
    filtered_bboxes = []
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
            filtered_confidences.append(confidences[i])

    return filtered_bboxes, filtered_confidences
