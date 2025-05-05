# Import necessary libraries
from ultralytics import YOLO  # YOLO object detection library
import pyzed.sl as sl  # ZED camera library
import numpy as np  # For numerical operations
import cv2  # OpenCV for image processing
import os  # For file and directory operations
import threading  # For handling keyboard input in a separate thread
import time  # For FPS calculation

# Import custom modules for detection and tracking
from DetectorDeepSort import YoloDetector
from TrackerDeepSort import Tracker

## Global variables
selected_object = 'q'  # Variable to store the ID of the selected object for tracking

# Define the path to the YOLO model
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "../Models/yolov8n.engine")

def init_zed():
    # Initialize the ZED camera with specific parameters
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Set depth mode to performance
    init_params.coordinate_units = sl.UNIT.METER  # Set units to meters
    init_params.sdk_verbose = 1  # Enable verbose logging
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Set camera resolution to HD720
    init_params.camera_fps = 30  # Set camera FPS to 30

    # Open the ZED camera and check for errors
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(err) + ". Exit program.")
        exit()
    return zed

def startup_message():
    # Display a startup message with instructions
    print("\nYOLO Object Detection with ZED on Jetson")
    print("Program started. Press 'q' in the window to stop.")
    print("Enter the ID of the object you want to track. Enter 'q' to stop tracking.")

class FPSCounter:
    def __init__(self):
        # Initialize FPS counter variables
        self.start_time = time.perf_counter()
        self.frame_count = 0

    def calculateFPS(self):
        # Calculate the frames per second (FPS)
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
        # Draw the FPS counter on the image
        text = f"FPS: {fps}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        text_color = (255, 255, 255)
        bg_color = (30, 30, 30)
        padding = 10

        # Get text size
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = 10, 30  # Top-left corner

        # Draw a rectangle and overlay the FPS text
        cv2.rectangle(img, (x - padding, y - h - padding), (x + w + padding, y + padding), bg_color, -1)
        cv2.putText(img, text, (x, y), font, scale, text_color, thickness)

class KeyboardThread(threading.Thread):
    def __init__(self, input_cbk=None, name='keyboard-input-thread'):
        # Initialize a thread to handle keyboard input
        self.input_cbk = input_cbk
        super(KeyboardThread, self).__init__(name=name, daemon=True)
        self.start()

    def run(self):
        # Continuously listen for keyboard input
        while True:
            self.input_cbk(input())

def my_callback(inp):
    # Handle keyboard input for selecting or stopping object tracking
    global selected_object
    if inp == 'q':
        print('Tracking stopped.', inp)
        selected_object = inp
    else:
        print('You are now tracking object:', inp)
        selected_object = inp

def calculateDistance(middle, depth_map):
    # Calculate the distance to the center of the object using the depth map
    depth_value = depth_map.get_value(middle[0], middle[1])
    distance = depth_value[1]
    return distance

def process_yolo_results(detections, tracking_ids, boxes, img_cv, depth_map):
    # Process YOLO detection results and draw bounding boxes
    global selected_object

    names = {
        0: 'person',
        1: 'car',
    }

    for tracking_id, bounding_box, detection in zip(tracking_ids, boxes, detections):
        x1, y1, x2, y2 = map(int, bounding_box)
        type = names.get(detection[1], "unknown")  # Get the object type
        confidence = round(detection[2], 2)  # Get the confidence score

        # Calculate the middle point of the object
        middle = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
        distance = calculateDistance(middle, depth_map)  # Calculate the distance

        # Draw bounding boxes with different colors based on selection
        if selected_object == str(tracking_id):
            draw_bounding_box(img_cv, x1, y1, x2, y2, (0, 0, 255), tracking_id, type, confidence, distance)
        else:
            draw_bounding_box(img_cv, x1, y1, x2, y2, (0, 255, 0), tracking_id, type, confidence, distance)

def draw_bounding_box(img_cv, x1, y1, x2, y2, color, tracking_id, type, confidence, distance):
    # Draw a bounding box and label on the image
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
    label_text = f"{tracking_id} {type} ({confidence:.2f}), distance: {distance:.2f} m"
    cv2.putText(img_cv, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main_loop(zed, detector, tracker, fps_counter, fps):
    # Main loop for processing frames and displaying results
    zed_image = sl.Mat()
    fixed_width = 1280
    fixed_height = 720

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the image from the ZED camera
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            img_cv = np.array(zed_image.get_data(), dtype=np.uint8)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)

            # Retrieve the depth map
            depth_map = sl.Mat()
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

            # Perform detection and tracking
            detections = detector.detect(img_cv)
            tracking_ids, boxes = tracker.track(detections, img_cv)

            # Process the results and draw bounding boxes
            process_yolo_results(detections, tracking_ids, boxes, img_cv, depth_map)

            # Update and display FPS
            fps_new, updated = fps_counter.calculateFPS()
            if updated:
                fps = fps_new
            fps_counter.draw_fps(img_cv, fps)

            # Resize and display the frame
            resized_frame = cv2.resize(img_cv, (fixed_width, fixed_height))
            cv2.imshow("YOLO Object Detection with ZED", resized_frame)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Close the ZED camera and destroy OpenCV windows
    zed.close()
    cv2.destroyAllWindows()

def main():
    # Main function to initialize components and start the program
    zed = init_zed()  # Initialize the ZED camera
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.70)  # Initialize YOLO detector
    tracker = Tracker()  # Initialize the tracker
    kthread = KeyboardThread(my_callback)  # Start the keyboard input thread
    fps_counter = FPSCounter()  # Initialize the FPS counter
    fps = 0  # Initialize FPS value
    startup_message()  # Display the startup message
    main_loop(zed, detector, tracker, fps_counter, fps)  # Start the main loop

if __name__ == "__main__":
    main()  # Run the main function
