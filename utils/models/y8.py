from ultralytics import YOLO


class myYOLO:
    def __init__(self):
        # self.model = YOLO("yolov8m-seg.pt")
        self.model = YOLO("yolov8m.pt")

    def get_bboxes(self, img):
        results = self.model.predict(img, classes=0)
        
        return results[0].boxes.xyxy
