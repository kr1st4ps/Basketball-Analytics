"""
Player utils.
"""


class Player:
    """
    Player class, containing all information of a detected and tracked player.
    """

    _id_counter = 0

    def __init__(self, start_frame, bbox):
        type(self)._id_counter += 1
        self.id = self._id_counter

        self.bbox_history = [bbox]

        self.start_frame = start_frame
        self.end_frame = None

    def __str__(self):
        return f"{self.id}.\tFrames[{self.start_frame}-{self.end_frame}]\tHistory{self.bbox_history}"

    def __del__(self):
        None

    def update(self, bbox):
        """
        Updates player info.
        """
        self.bbox_history.insert(0, bbox)
        if len(self.bbox_history) > 5:
            self.bbox_history.pop()

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
