from ultralytics import YOLO


class YoloDetector:
  def __init__(self, model_path, confidence):
    self.model = YOLO(model_path)
    self.classList = ["person"]
    self.confidence = confidence

  def detect(self, image):
    results = self.model.predict(image, conf=self.confidence, iou=0.4, verbose=False)
    result = results[0]
    detections = self.make_detections(result)
    return detections

  def make_detections(self, result):
    boxes = result.boxes
    detections = []
    for box in boxes:
      x1, y1, x2, y2 = map(int, box.xyxy[0])
      w, h = x2 - x1, y2 - y1
      class_number = int(box.cls[0])

      if result.names[class_number] not in self.classList:
        continue
      conf = box.conf.item()
      detections.append((([x1, y1, w, h]), class_number, conf))
    return detections