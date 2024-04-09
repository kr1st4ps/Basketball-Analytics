"""
Actions with detectron2 models.
"""

import time
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances


class myDetectron:
    """
    detectron2 model class.
    """

    def __init__(self, version):
        self.cfg = get_cfg()
        if version == "big":
            self.cfg.merge_from_file(
                "/Users/kristapsalmanis/projects/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
            )
            self.cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"
        elif version == "small":
            self.cfg.merge_from_file(
                "/Users/kristapsalmanis/projects/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
            )
            self.cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(self.cfg)

    def get_shapes(self, img, id):
        """
        Gets all shapes.
        """
        start_time = time.time()
        outputs = self.predictor(img)
        end_time = time.time()
        inference_time = end_time - start_time
        print("Inference time: ", inference_time, "seconds")

        return filter_predictions_by_id(outputs, id, img.shape[:2])

    def get_annotated_image(self, img, outputs):
        """
        Draws detections on image.
        """
        v = Visualizer(
            img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return v.get_image()[:, :, ::-1]


def filter_predictions_by_id(outputs, id, img_size):
    """
    Filters detections by id.
    """
    instances = outputs["instances"]

    mask = instances.pred_classes == id
    pred_masks = instances.pred_masks[mask]
    pred_classes = instances.pred_classes[mask]
    scores = instances.scores[mask]
    pred_boxes = instances.pred_boxes[mask]

    filtered_instance = Instances(image_size=(img_size))
    filtered_instance.set("pred_classes", pred_classes)
    filtered_instance.set("pred_boxes", pred_boxes)
    filtered_instance.set("pred_masks", pred_masks)
    filtered_instance.set("scores", scores)
    filtered_output = {"instances": filtered_instance}

    return filtered_output
