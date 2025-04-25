from ultralytics import YOLO
import pyzed.sl as sl
import numpy as np
import cv2
import os
import threading
import time

from DetectorDeepSort import YoloDetector
from TrackerDeepSort import Tracker

## Global variables
selected_object = 'q'

fixed_camera = False




script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "../Models/yolov8s.pt")

def init_zed():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

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

def process_yolo_results(detections, tracking_ids, boxes, img_cv):
    global selected_object

    names = {
        0: 'person',
        1: 'car',
    }

    for tracking_id, bounding_box, detection in zip(tracking_ids, boxes, detections):
        x1, y1, x2, y2 = map(int, bounding_box)
        type = names.get(detection[1], "unknown")
        confidence = round(detection[2], 2)

        if selected_object == str(tracking_id):
            draw_bounding_box(img_cv, x1, y1, x2, y2, (0, 0, 255), tracking_id, type, confidence)
        else:
            draw_bounding_box(img_cv, x1, y1, x2, y2, (0, 255, 0), tracking_id, type, confidence)

def draw_bounding_box(img_cv, x1, y1, x2, y2, color, tracking_id, type, confidence):
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
    label_text = f"{tracking_id} {type} ({confidence:.2f})"
    cv2.putText(img_cv, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def startup_message():
    print("\nYOLO Object Detection with ZED on Jetson")
    print("Program started. Press 'q' in the terminal or window to stop.")
    print("Enter the ID of the object you want to track. Enter 'q' to stop tracking.")

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


def main_loop(zed, detector, tracker, fps_counter, fps):
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

            process_yolo_results(detections, tracking_ids, boxes, img_cv)

            fps_new, updated = fps_counter.calculateFPS()
            if updated:
                fps = fps_new
            fps_counter.draw_fps(img_cv, fps)

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
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.70)
    tracker = Tracker()
    kthread = KeyboardThread(my_callback)
    fps_counter = FPSCounter()
    fps = 0
    main_loop(zed, detector, tracker, fps_counter, fps)

if __name__ == "__main__":
    main()
