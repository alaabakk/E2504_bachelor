from ultralytics import YOLO


class YoloDetector:
  def __init__(self, model_path, confidence):
    # Initialize the YOLO model with the specified model path and detection task
    self.model = YOLO(model_path, task="detect")
    # Define the list of classes to detect (e.g., "person")
    self.classList = ["person"]
    # Set the confidence threshold for detections
    self.confidence = confidence

  def detect(self, image):
    # Perform object detection on the input image
    results = self.model.predict(image, conf=self.confidence, iou=0.4, verbose=False)
    # Get the first result from the predictions
    result = results[0]
    # Process the result to extract detections
    detections = self.make_detections(result)
    return detections

  def make_detections(self, result):
    # Extract bounding boxes from the detection result
    boxes = result.boxes
    detections = []  # List to store processed detections
    for box in boxes:
      # Extract bounding box coordinates and convert them to integers
      x1, y1, x2, y2 = map(int, box.xyxy[0])
      w, h = x2 - x1, y2 - y1  # Calculate width and height of the bounding box
      # Get the class number of the detected object
      class_number = int(box.cls[0])

      # Skip detections that are not in the specified class list
      if result.names[class_number] not in self.classList:
        continue
      # Get the confidence score of the detection
      conf = box.conf.item()
      # Append the detection as a tuple (bounding box, class number, confidence)
      detections.append((([x1, y1, w, h]), class_number, conf))
    return detections