from ultralytics import YOLO

# Load model
model = YOLO("yolov8x.pt")

model.export(format='engine', device=0, half=True)  # export to TensorRT engine
