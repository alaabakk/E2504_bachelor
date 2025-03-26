from ultralytics import YOLO
import pyzed.sl as sl
import numpy as np
import cv2
import os
import threading
import Jetson.GPIO as GPIO

from DetectorDeepSort import YoloDetector
from TrackerDeepSort import Tracker

## Global variables
active_objects = []
last_active_objects = []
selected_object = 'q'

fixed_camera = False
GPIO.setmode(GPIO.BOARD)
servoPin1 = 32
servoPin2 = 33

script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the model file
MODEL_PATH = os.path.join(script_dir, "Models/yolov8s.engine")

def init_zed():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1
    init_params.camera_resolution = sl.RESOLUTION.HD720

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(err) + ". Exit program.")
        exit()

    return zed

class KeyboardThread(threading.Thread):

    def __init__(self, input_cbk = None, name='keyboard-input-thread'):
        self.input_cbk = input_cbk
        super(KeyboardThread, self).__init__(name=name, daemon=True)
        self.start()

    def run(self):
        while True:
            self.input_cbk(input()) #waits to get input + Return

def my_callback(inp):
    global selected_object
    #evaluate the keyboard input
    if inp == 'q':
        print('Tracking stopped.', inp)
        selected_object = inp
    else:
        print('Your now tracking object:', inp)
        selected_object = inp

def process_yolo_results(detections, tracking_ids, boxes, img_cv, servo1, servo2):
    global active_objects
    active_objects = []
    global selected_object

    # Define the classes to keep
    names = {
        0: 'person',
        1: 'car',
        2: 'motorcycle',
        4: 'bus',
        6: 'truck',
    }

    for tracking_id, bounding_box,  detection in zip(tracking_ids, boxes, detections):
        x1, y1, x2, y2 = bounding_box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        type = names[detection[1]]  # Get class label from the dictionary
        active_objects.append([tracking_id, type])
        confidence = round(detection[2], 2)

        if selected_object == str(tracking_id):
            # Draw bounding box
            draw_bounding_box(img_cv, x1, y1, x2, y2, (0, 0, 255), tracking_id, type, confidence)
            # Control the servo
            servo_control(x1, y1, x2, y2, servo1, servo2)

        else:
            # Draw bounding box
            draw_bounding_box(img_cv, x1, y1, x2, y2, (0, 255, 0), tracking_id, type, confidence)

            # Check if no object is selected
            if selected_object == 'q':
                # Control the servo
                servo_control(x1, y1, x2, y2, servo1, servo2)


    return active_objects



def print_active_objects(active_objects):
    global last_active_objects
    if active_objects != last_active_objects:
        if active_objects:
            print("\nActive Objects:")
            print(f"{'ID':<10}{'Type':<15}")
            print("-" * 25)
            for obj in active_objects:
                print(f"{obj[0]:<10}{obj[1]:<15}")
        else:
            print("\nNo active objects detected.")

    last_active_objects = active_objects

def draw_bounding_box(img_cv, x1, y1, x2, y2, color, tracking_id, type, confidence):
    # Draw bounding box
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
    # Add label and confidence
    label_text = f"{tracking_id} {type} ({confidence:.2f})"  # Updated to show object name
    cv2.putText(img_cv, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def servo_control(x1, y1, x2, y2, servo1, servo2):
    global selected_object
    # Calculate the center of the bounding box
    x_center = ((x2 - x1) / 2) + x1
    y_center = ((y2 - y1) / 2) + y1
    # Calculate the angle from 35 to 145 degrees
    angle_x = 35 + (x_center / 1280) * (145 - 35)
    angle_y = 35 + (y_center / 720) * (145 - 35)

    
    if selected_object == 'q':
        # Set the servo to the center (DutyCycle: 7.5 == 90 degrees)
        servo1.ChangeDutyCycle(7.5)
        servo2.ChangeDutyCycle(7.5)

    else:
        # 180 - to invert the direction of servos
        angle_x = 180 - max(0, min(180, angle_x))
        angle_y = 180 - max(0, min(180, angle_y))

        min_duty = 2.5
        max_duty = 12.5
        duty_x = min_duty + (angle_x / 180.0) * (max_duty - min_duty)
        duty_y = min_duty + (angle_y / 180.0) * (max_duty - min_duty)

        servo1.ChangeDutyCycle(duty_x)
        servo2.ChangeDutyCycle(duty_y)
    

def startup_message():
    # Start up information
    print("\n \nYOLO Object Detection with ZED")
    print("Program started. Press 'q' in the video window to stop the program.")
    print("Enter the ID of the object you want to track, enter q to stop tracking.")


def main_loop(zed, detector, tracker, servo1, servo2):
    # Create a ZED Mat object to store images
    zed_image = sl.Mat()

    # Define fixed width and height
    fixed_width = 1280
    fixed_height = 720

    first_iteration = True

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image from the ZED camera
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            img_cv = np.array(zed_image.get_data(), dtype=np.uint8)

            # Convert RGBA to RGB
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)

            # Predict using YOLO
            detections = detector.detect(img_cv)
            tracking_ids, boxes = tracker.track(detections, img_cv)

            # Process results and draw on the frame
            process_yolo_results(detections, tracking_ids, boxes, img_cv, servo1, servo2)


            # Resize the frame to fixed dimensions
            resized_frame = cv2.resize(img_cv, (fixed_width, fixed_height))

            # Display the resized frame
            cv2.imshow("YOLO Object Detection with ZED", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if first_iteration:
                startup_message()
                first_iteration = False

    # Sets servo to center position and stops the program
    servo1.ChangeDutyCycle(7.5)
    servo2.ChangeDutyCycle(7.5)
    
    zed.close()
    cv2.destroyAllWindows()
    servo1.stop
    servo2.stop
    GPIO.cleanup()


def main():
    # Initialize ZED camera
    zed = init_zed()

    # Initialize YOLO detector and tracker
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.75)
    tracker = Tracker()

    # Start the keyboard input thread
    kthread = KeyboardThread(my_callback)

    # Pin 32 = pwmchip3/pwm0
    servo1 = GPIO.PWM(servoPin1, 50)
    # Pin 33 = pwmchip0/pwm0
    servo2 = GPIO.PWM(servoPin2, 50)

    servo1.start(7.5)
    servo2.start(7.5)

    # Start the main loop
    main_loop(zed, detector, tracker, servo1, servo2)


if __name__ == "__main__":
    main()
