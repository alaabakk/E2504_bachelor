import cv2
import time
from DetectorDeepSort import YoloDetector
from TrackerDeepSort import Tracker
import pyzed.sl as sl
import numpy as np

MODEL_PATH = "Models/yolov5nu.pt"

def init_zed():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_fps = 10
    init_params.sdk_verbose = 1
    init_params.camera_resolution = sl.RESOLUTION.HD720

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(err) + ". Exit program.")
        exit()

    return zed

def main():
    # Initialize ZED camera
    zed = init_zed()

    # Initialize YOLO detector and tracker
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.2)
    tracker = Tracker()

    # Create a ZED Mat object to store images
    zed_image = sl.Mat()

    frame_counter=0

    while True:

        # Grab a new frame from the ZED camera
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image from the ZED camera
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            frame = np.array(zed_image.get_data(), dtype=np.uint8)

            # Convert RGBA to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            # Increment the frame counter
            frame_counter += 1
            start_time = time.perf_counter()
            # Only process every third frame
            if True:
                # Perform detection and tracking
                start_time = time.perf_counter()
                detections = detector.detect(frame)
                tracking_ids, boxes = tracker.track(detections, frame)

                # Draw bounding boxes and tracking IDs
                for tracking_id, bounding_box in zip(tracking_ids, boxes):
                    cv2.rectangle(frame, (int(bounding_box[0]), int(bounding_box[1])), 
                                  (int(bounding_box[2]), int(bounding_box[3])), (0, 0, 255), 2)
                    cv2.putText(frame, f"{str(tracking_id)}", 
                                (int(bounding_box[0]), int(bounding_box[1] - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                

                

            # Calculate and display FPS
            end_time = time.perf_counter()
            fps = 1 / (end_time - start_time)
            print(f"Current fps: {fps}")

            # Display the frame
            cv2.imshow("Frame", frame)

            # Break the loop if 'q' or 'Esc' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    # Release resources
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()