from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import detectron2
import cv2
import numpy as np
import torch
from detectron2.structures import Instances
import supervision as sv
from shapely.geometry import Polygon, Point



class myDetectron:
    def __init__(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file("/Users/kristapsalmanis/projects/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"
        self.cfg.MODEL.DEVICE = "cpu"
        
        self.predictor = DefaultPredictor(self.cfg)

    def get_shapes(self, img, id):
        outputs = self.predictor(img)

        return filter_predictions_by_id(outputs, id, img.shape[:2])
    
    def get_annotated_image(self, img, outputs):
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return v.get_image()[:, :, ::-1]
    

def filter_predictions_by_id(outputs, id, img_size):
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

def filter_predictions_by_court(outputs, court, img_size):
    instances = outputs["instances"]

    court_polygon = Polygon(court)

    indices_in_court = []
    for box in instances.pred_boxes.tensor:
        bottom_center = ((box[0] + box[2]) / 2, box[3])

        if Point(bottom_center).within(court_polygon):
            indices_in_court.append(True)
        else:
            indices_in_court.append(False)

    mask = torch.tensor(indices_in_court)

    pred_masks = instances.pred_masks[mask]
    pred_classes = instances.pred_classes[mask]
    scores = instances.scores[mask]
    pred_boxes = instances.pred_boxes[mask]

    filtered_instance = Instances(image_size=img_size)
    filtered_instance.set("pred_classes", pred_classes)
    filtered_instance.set("pred_boxes", pred_boxes)
    filtered_instance.set("pred_masks", pred_masks)
    filtered_instance.set("scores", scores)

    filtered_output = {"instances": filtered_instance}

    return filtered_output


