from ultralytics import YOLO

# Load your model (change path if needed)
model = YOLO('yolov8n.pt')  # or 'path/to/your/model.pt'

# Evaluate the model on your dataset
metrics = model.val(data='/Users/olefjeldhaugstvedt/Dokumenter Lokal/E2504_bachelor/val/data.yaml', imgsz=640, batch=16)

# Print results
print(metrics)
