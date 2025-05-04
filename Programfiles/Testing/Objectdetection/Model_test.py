from ultralytics import YOLO
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '../../../Models/yolov8s.pt')
# Load your model (change path if needed)
model = YOLO(model_path)  # or 'path/to/your/model.pt'

# Add path to dataset.yaml
data_path = os.path.join(script_dir, 'Testset/Cross-path.v1i.yolov8/data.yaml')
# Evaluate the model on your dataset
metrics = model.val(data=data_path, imgsz=640, batch=16, split="test", half=True, classes = [0],)

#print metrics
precision_class0 = metrics.box.p[0]
recall_class0 = metrics.box.r[0]
f1_class0 = metrics.box.f1[0]
map_class0 = metrics.box.maps[0]

print(f"Precision (class 0): {precision_class0:.4f}")
print(f"Recall (class 0):    {recall_class0:.4f}")
print(f"F1-score (class 0):  {f1_class0:.4f}")
print(f"mAP@0.5:0.95 (class 0): {map_class0:.4f}")
