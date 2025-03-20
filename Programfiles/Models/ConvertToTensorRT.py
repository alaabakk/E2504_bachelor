from ultralytics import YOLO

# Load model
model = YOLO("yolov8s.pt")

model.export(format='engine', device=0)  # export to TensorRT engine