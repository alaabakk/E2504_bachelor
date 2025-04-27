from ultralytics import YOLO
import pyzed.sl as sl
import numpy as np
import cv2
import os

import time

start_time, frame_count = time.time(), 0

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
    model_path = os.path.join(script_dir, "../../../Models/yolov8n.engine")
    model = YOLO(model_path, task="detect")
    print("YOLO model initialized")
    return model 


def calculateFPS():
    # Calculate the FPS based on the time taken to process each frame
    global start_time, frame_count
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        fps = int(fps)
        frame_count = 0
        start_time = current_time
        return fps, True

    frame_count += 1
    return 0, False



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

                

def main_loop(zed, model):
    # Create a ZED Mat object to store images
    zed_image = sl.Mat()


    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image from the ZED camera
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            img_cv = np.array(zed_image.get_data(), dtype=np.uint8)

            # Convert RGBA to RGB
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
            img_cv = cv2.resize(img_cv, (640, 640))

            # Predict using YOLO
            results = model.predict(img_cv, stream=True, conf=0.7, verbose = True)

            # Process results and draw on the frame
            process_yolo_results(results, img_cv)

            # Calculate FPS
            fps_new, state = calculateFPS()
            if state:
                fps = fps_new

            cv2.rectangle(img_cv, (0, 0), (100, 50), (0, 0, 0), -1)
            cv2.putText(img_cv, str(fps), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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

    # Start the main loop
    main_loop(zed, model)


if __name__ == "__main__":
    main()
