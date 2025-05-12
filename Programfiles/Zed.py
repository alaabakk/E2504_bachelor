import pyzed.sl as sl
import numpy as np
import cv2
import time

import os
import threading

## Global variables  
selected_object = 'q' 


def init_zed():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(err)+". Exit program.")
        exit()

    return zed
    
def init_object_detection(zed):
    # Object detection configuration
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking=True
    obj_param.enable_segmentation=True
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST

    if obj_param.enable_tracking :
        positional_tracking_param = sl.PositionalTrackingParameters()
        zed.enable_positional_tracking(positional_tracking_param)

    print("Object Detection: Loading Module...")

    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS :
        print("Enable object detection : "+repr(err)+". Exit program.")
        zed.close()
        exit()

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 60

    return obj_param, obj_runtime_param

# Samler alle initialiseringsfunksjonene i en funksjon 
def init():
    zed = init_zed()
    print("ZED initialized")
    _, obj_runtime_param = init_object_detection(zed)
    print("Object detection initialized")
    kthread = KeyboardThread(input_cbk=my_callback)
    print("Keyboard thread initialized")
    

    fps_counter = FPSCounter()
    fps = 0

    return zed, obj_runtime_param, fps_counter, fps

def startup_message():
    print("\nYOLO Object Detection with ZED on Jetson")
    print("Program started. Press 'q' in the window to stop.")
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
    print('You selected object:', inp)
    global selected_object
    selected_object = inp


def calculateDistance(middle, depth_map):
    # Get the depth value at the center of the object
    depth_value = depth_map.get_value(middle[0], middle[1])
    distance = depth_value[1]

    return distance


def process_objects(objects, img_cv, depth_map):
    global selected_object
    
    # Define the classes to keep
    names = [
        'PERSON',
        'CAR',
    ]

    if objects.is_new:
        obj_array = objects.object_list

        for obj in obj_array:
            if str(obj.label) in names: # Filter for the class "PERSON"
                x1, y1, x2, y2 = int(obj.bounding_box_2d[0][0]), int(obj.bounding_box_2d[0][1]), int(obj.bounding_box_2d[2][0]), int(obj.bounding_box_2d[2][1])

                # Avstand til midten av objektet
                middle = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
                distance = calculateDistance(middle, depth_map)

                if selected_object == str(obj.id):
                    draw_bounding_box(img_cv, x1, y1, x2, y2, (0, 0, 255), obj.id, obj.label, obj.confidence, distance)

                else:
                    draw_bounding_box(img_cv, x1, y1, x2, y2, (0, 255, 0), obj.id, obj.label, obj.confidence, distance)

def draw_bounding_box(img_cv, x1, y1, x2, y2, color, tracking_id, type, confidence, distance):
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
    label_text = f"{tracking_id} {type} ({confidence:.2f}), distance: {distance:.2f} m"
    cv2.putText(img_cv, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



def main_loop(zed, obj_runtime_param, fps_counter, fps):
    objects = sl.Objects()
    
    
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            img = sl.Mat()
            zed.retrieve_image(img, sl.VIEW.LEFT)
            img_cv = np.array(img.get_data(), dtype=np.uint8)

            zed.retrieve_objects(objects, obj_runtime_param)

            #distanse
            depth_map = sl.Mat()
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            
            process_objects(objects, img_cv, depth_map)

            # Calculate and draw FPS
            fps_new, updated = fps_counter.calculateFPS()
            if updated:
                fps = fps_new
            fps_counter.draw_fps(img_cv, fps)


            cv2.imshow("Object detection with ZED", img_cv)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.disable_object_detection()
    zed.close()
    cv2.destroyAllWindows()


def main():
    zed, obj_runtime_param, fps_counter, fps = init()
    startup_message
    main_loop(zed, obj_runtime_param, fps_counter, fps)


if __name__ == "__main__":
    main()
