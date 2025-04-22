from ultralytics import YOLO

# Load model
model = YOLO("yolo8.s.1.pt")

model.export(format='engine', device=0)  # export to TensorRT engine