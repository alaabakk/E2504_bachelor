from ultralytics import YOLO

# Load model
model = YOLO("customYOLOv8s_1.pt")

model.export(format='engine', device=0)  # export to TensorRT engine