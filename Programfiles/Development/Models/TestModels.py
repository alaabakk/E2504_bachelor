from ultralytics import YOLO
 
 
#print("Starting to predict")
model = YOLO("yolov5n.pt", task="detect")
 
 
results = model.predict("people-metro-station.mp4", save=True, stream=True, conf=0.5, iou=0.4)
#print("Prediction done")
for r in results:
    print(r.summary())