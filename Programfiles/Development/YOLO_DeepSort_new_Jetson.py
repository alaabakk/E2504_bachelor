from ultralytics import YOLO
import pyzed.sl as sl
import numpy as np
import cv2
import os
import threading
import serial
import time

from DetectorDeepSort import YoloDetector
from TrackerDeepSort import Tracker

## Global variables
active_objects = []
last_active_objects = []
selected_object = 'q'

fixed_camera = False

## Serial configuration
PORT = '/dev/ttyUSB0'  # <-- Use known USB port
BAUDRATE = 115200
TIMEOUT = 1

def init_serial():
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=TIMEOUT)
        time.sleep(2)
        print(f"Serial port {PORT} opened successfully.")
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port {PORT}: {e}")
        return None


script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "Models/yolov8s.engine")

def init_zed():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1
    init_params.camera_resolution = sl.RESOLUTION.HD720

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(err) + ". Exit program.")
        exit()
    return zed

class KeyboardThread(threading.Thread):
    def __init__(self, input_cbk=None, name='keyboard-input-thread'):
        self.input_cbk = input_cbk
        super(KeyboardThread, self).__init__(name=name, daemon=True)
        self.start()

    def run(self):
        while True:
            self.input_cbk(input())

def my_callback(inp):
    global selected_object
    if inp == 'q':
        print('Tracking stopped.', inp)
        selected_object = inp
    else:
        print('You are now tracking object:', inp)
        selected_object = inp

def process_yolo_results(detections, tracking_ids, boxes, img_cv, ser):
    global active_objects
    active_objects = []
    global selected_object

    names = {
        0: 'person',
        1: 'car',
        2: 'motorcycle',
        4: 'bus',
        6: 'truck',
    }

    for tracking_id, bounding_box, detection in zip(tracking_ids, boxes, detections):
        x1, y1, x2, y2 = map(int, bounding_box)
        type = names.get(detection[1], "unknown")
        active_objects.append([tracking_id, type])
        confidence = round(detection[2], 2)

        if selected_object == str(tracking_id):
            draw_bounding_box(img_cv, x1, y1, x2, y2, (0, 0, 255), tracking_id, type, confidence)
            serial_print(ser, x1, y1, x2, y2)
        else:
            draw_bounding_box(img_cv, x1, y1, x2, y2, (0, 255, 0), tracking_id, type, confidence)

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
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
    label_text = f"{tracking_id} {type} ({confidence:.2f})"
    cv2.putText(img_cv, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def serial_print(ser, x1, y1, x2, y2):
    if not ser:
        return
    message1 = 640 - (x1 + x2) / 2
    message2 = 360 - (y1 + y2) / 2
    message = f"{message1} , {message2}\n"
    ser.write(message.encode())

def startup_message():
    print("\nYOLO Object Detection with ZED on Jetson")
    print("Program started. Press 'q' in the terminal or window to stop.")
    print("Enter the ID of the object you want to track. Enter 'q' to stop tracking.")

def main_loop(zed, detector, tracker, ser):
    zed_image = sl.Mat()
    fixed_width = 1280
    fixed_height = 720
    first_iteration = True

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            img_cv = np.array(zed_image.get_data(), dtype=np.uint8)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)

            detections = detector.detect(img_cv)
            tracking_ids, boxes = tracker.track(detections, img_cv)

            process_yolo_results(detections, tracking_ids, boxes, img_cv, ser)

            resized_frame = cv2.resize(img_cv, (fixed_width, fixed_height))
            cv2.imshow("YOLO Object Detection with ZED", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if first_iteration:
                startup_message()
                first_iteration = False

    zed.close()
    cv2.destroyAllWindows()

def main():
    zed = init_zed()
    ser = init_serial()
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.75)
    tracker = Tracker()
    kthread = KeyboardThread(my_callback)
    main_loop(zed, detector, tracker, ser)

if __name__ == "__main__":
    main()
