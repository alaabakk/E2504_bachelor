from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../../../Models/yolov8n.engine")
# Load your model (change path if needed)
model = YOLO(model_path)  # or 'path/to/your/model.pt'

# Add path to dataset.yaml
data_path = os.path.join(script_dir, "/Testset/Distance.v1i.yolov8/data.yaml")
# Evaluate the model on your dataset
metrics = model.val(data_path, imgsz=640, batch=16, split="test", classes = [0],)

# Print results
print(metrics)

f1_class0 = metrics.box.f1[0]
print(f"F1-score for class 0 (person): {f1_class0:.4f}")