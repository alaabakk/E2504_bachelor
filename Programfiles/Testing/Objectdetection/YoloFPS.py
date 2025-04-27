from ultralytics import YOLO
import pyzed.sl as sl
import numpy as np
import cv2
import os

import time



def init_zed():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30


    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(err) + ". Exit program.")
        exit()

    return zed


def init_yolo():
    # Initialize the YOLO model
    print("Initializing YOLO model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the model file
    model_path = os.path.join(script_dir, "../../../Models/yolov8s.pt")
    model = YOLO(model_path, task="detect")
    print("YOLO model initialized")
    return model 

class FPSCounter:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.frame_count = 0

    def calculateFPS(self):
        current_time = time.perf_counter()
        elapsed_time = current_time - self.start_time

        self.frame_count += 1

        if elapsed_time > 1.0:
            fps = int(self.frame_count / elapsed_time)
            self.start_time = current_time
            self.frame_count = 0
            return fps, True
        
        return 0, False
    
    def draw_fps(self, img, fps):
        # Set styles
        text = f"FPS: {fps}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        text_color = (255, 255, 255)
        bg_color = (30, 30, 30)  # Dark grey
        padding = 10

        # Get text size
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = 10, 30  # Top-left corner

        # Draw rounded rectangle (you can swap to plain if preferred)
        cv2.rectangle(img, (x - padding, y - h - padding), (x + w + padding, y + padding), bg_color, -1)
        cv2.putText(img, text, (x, y), font, scale, text_color, thickness)



def process_yolo_results(results, img_cv):
    # Define the classes to keep
    names = {
        0: 'person',
    }

    # Process YOLO results and draw bounding boxes on the image
    for r in results:
        for box in r.boxes:
            label_id = int(box.cls[0])  # Class ID
            if label_id in names:  # Filter by desired classes
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0]  # Confidence score
                label = names[label_id]  # Get class label from the dictionary

                # Draw bounding box
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label and confidence
                label_text = f"{label} ({confidence:.2f})"  # Updated to show object name
                cv2.putText(img_cv, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                

def main_loop(zed, model, fps_counter):
    # Create a ZED Mat object to store images
    zed_image = sl.Mat()

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image from the ZED camera
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            img_cv = np.array(zed_image.get_data(), dtype=np.uint8)

            # Convert RGBA to RGB
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)

            # Predict using YOLO
            results = model.predict(img_cv, stream=True, conf=0.7)

            # Process results and draw on the frame
            process_yolo_results(results, img_cv)


            fps_new, updated = fps_counter.calculateFPS()
            if updated:
                fps = fps_new
            fps_counter.draw_fps(img_cv, fps)


            # Display the frame
            cv2.imshow("YOLO Object Detection with ZED", img_cv)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()


def main():
    # Initialize ZED camera
    zed = init_zed()

    # Initialize YOLO model
    model = init_yolo()

    # Create a FPSCounter object
    fps_counter = FPSCounter()

    # Start the main loop
    main_loop(zed, model, fps_counter)


if __name__ == "__main__":
    main()
