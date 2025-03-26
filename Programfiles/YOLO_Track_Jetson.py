from ultralytics import YOLO
import pyzed.sl as sl
import numpy as np
import cv2
import os
import threading
import Jetson.GPIO as GPIO
import time

## Global variables
active_objects = []
last_active_objects = []
selected_object = None

fixed_camera = False
GPIO.setmode(GPIO.BOARD)
servoPin1 = 32
servoPin2 = 33

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

def init_yolo():
    # Initialize the YOLO model
    print("Initializing YOLO model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the model file
    model_path = os.path.join(script_dir, "yolov8s.engine")
    model = YOLO(model_path, task="detect")
    print("YOLO model initialized")
    return model

class KeyboardThread(threading.Thread):

    def __init__(self, input_cbk = None, name='keyboard-input-thread'):
        self.input_cbk = input_cbk
        super(KeyboardThread, self).__init__(name=name, daemon=True)
        self.start()

    def run(self):
        while True:
            self.input_cbk(input()) #waits to get input + Return

def my_callback(inp):
    #evaluate the keyboard input
    print('Enter the ID of the object you want to track, enter q to stop tracking:', inp)
    global selected_object
    selected_object = inp

def process_yolo_results(results, img_cv, servo1, servo2):
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

    # Process YOLO results and draw bounding boxes on the image
    for r in results:
        for box in r.boxes:
            label_id = int(box.cls[0])  # Class ID
            if label_id in names:  # Filter by desired classes
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0]  # Confidence score
                type = names[label_id]  # Get class label from the dictionary
                ID = int(box.id[0])  # Get the unique ID of the object

                active_objects.append([ID, type]) # Add object to the list of active objects

                if selected_object == str(ID):
                    # Draw bounding box
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # Add label and confidence
                    label_text = f"{ID} {type} ({confidence:.2f})"  # Updated to show object name
                    cv2.putText(img_cv, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Control the servo
                    servo_control(x1, y1, x2, y2, servo1, servo2)


                else:
                    # Draw bounding box
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Add label and confidence
                    label_text = f"{ID} {type} ({confidence:.2f})"  # Updated to show object name
                    cv2.putText(img_cv, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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



def servo_control(x1, y1, x2, y2, servo1, servo2):
    global last_angle_x
    global last_angle_y
    # Calculate the center of the bounding box
    x_center = ((x2 - x1) / 2) + x1
    y_center = ((y2 - y1) / 2) + y1
    # Calculate the angle from center(90) to the object
    delta_x = (x_center - 640) / 640 * 55
    delta_y = (y_center - 360) / 360 * 55

    if fixed_camera == True:     
        pass

    elif fixed_camera == False:
        # Calculate the actual servo position
        actual_angle_x = 90 + delta_x
        actual_angle_y = 90 + delta_y
        # 180 - to invert the direction of servos
        actual_angle_x = 180 - max(0, min(180, actual_angle_x))
        actual_angle_y = 180 - max(0, min(180, actual_angle_y))

        min_duty = 2.5
        max_duty = 12.5
        duty_x = min_duty + (actual_angle_x / 180.0) * (max_duty - min_duty)
        duty_y = min_duty + (actual_angle_y / 180.0) * (max_duty - min_duty)

        servo1.ChangeDutyCycle(duty_x)
        servo2.ChangeDutyCycle(duty_y)


def main_loop(zed, model, servo1, servo2):
    # Create a ZED Mat object to store images
    zed_image = sl.Mat()

    # Define fixed width and height
    fixed_width = 1280
    fixed_height = 720

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image from the ZED camera
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            img_cv = np.array(zed_image.get_data(), dtype=np.uint8)

            # Convert RGBA to RGB
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)

            # Predict using YOLO
            results = model.track(img_cv, stream=True, augment=True, verbose=False)

            # Process results and draw on the frame
            active_objects = process_yolo_results(results, img_cv, servo1, servo2)

            #print_active_objects(active_objects)

            # Resize the frame to fixed dimensions
            resized_frame = cv2.resize(img_cv, (fixed_width, fixed_height))

            # Display the resized frame
            cv2.imshow("YOLO Object Detection with ZED", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()
    servo1.stop
    servo2.stop
    GPIO.cleanup()


def main():
    # Initialize ZED camera
    zed = init_zed()

    # Initialize YOLO model
    model = init_yolo()

    # Start the keyboard input thread
    kthread = KeyboardThread(my_callback)

    # Pin 32 = pwmchip3/pwm0
    servo1 = GPIO.PWM(servoPin1, 50)
    # Pin 33 = pwmchip0/pwm0
    servo2 = GPIO.PWM(servoPin2, 50)

    servo1.start(7.5)
    servo2.start(7.5)

    # Start the main loop
    main_loop(zed, model, servo1, servo2)


if __name__ == "__main__":
    main()
