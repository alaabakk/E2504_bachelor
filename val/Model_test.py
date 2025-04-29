from ultralytics import YOLO

# Load your model (change path if needed)
model = YOLO('yolov8n.pt')  # or 'path/to/your/model.pt'

# Evaluate the model on your dataset
<<<<<<< HEAD
metrics = model.val(data=r'C:\Users\Pette\OneDrive\Dokumenter\Skole\Bachelor\Code\E2504_bachelor\val\data.yaml', imgsz=640, batch=16)
=======
metrics = model.val(data='/Users/olefjeldhaugstvedt/Dokumenter Lokal/E2504_bachelor/val/data.yaml', imgsz=640, batch=16)
>>>>>>> 611ec0f21edb7966fa0da4e99de0dc421f5a4221

# Print results
print(metrics)
