from ultralytics import YOLO

# Load your model (change path if needed)
model = YOLO("yolov8n.pt")  # or 'path/to/your/model.pt'

# Evaluate the model on your dataset
metrics = model.val(data=r'C:\Users\Pette\OneDrive\Dokumenter\Skole\Bachelor\Code\Distance.v1i.yolov8\data.yaml', imgsz=640, batch=16, split="test", half=True, classes = [0],)

# Print results
print(metrics)

map_class0 = metrics.box.maps[0]  # for mAP@0.5
print(f"mAP@0.5 for class 0 (person): {map_class0:.4f}")
print()

f1_class0 = metrics.box.f1[0]
print(f"F1-score for class 0 (person): {f1_class0:.4f}")
