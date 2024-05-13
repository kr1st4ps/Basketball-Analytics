from ultralytics import YOLO

model = YOLO("yolov8x-seg.pt")

results = model(
    "resources/images/test_img.png", save=True, show_labels=False, show_conf=False
)
