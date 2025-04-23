import pyzed.sl as sl
import numpy as np
import cv2
import time

import os
import threading

## Global variables  
selected_object = None 
start_time, frame_count = time.time(), 0




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
    obj_param.enable_tracking=False
    obj_param.enable_segmentation=True
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE

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


# Samler alle initialiseringsfunksjonene i en funksjon 
def init():
    zed = init_zed()
    print("ZED initialized")
    _, obj_runtime_param = init_object_detection(zed)
    print("Object detection initialized")
    kthread = KeyboardThread(input_cbk=my_callback)
    print("Keyboard thread initialized")

    return zed, obj_runtime_param


def calculateFPS():
    # Calculate the FPS based on the time taken to process each frame
    global start_time, frame_count
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = current_time

    frame_count += 1


def calculateDistance(middle, depth_map):
    # Get the depth value at the center of the object
    depth_value = depth_map.get_value(middle[0], middle[1])
    distance = depth_value[1]

    return distance


def process_objects(objects, img_cv, depth_map):
    global selected_object


    if objects.is_new:
        obj_array = objects.object_list
        #print(f"{len(obj_array)} Object(s) detected")

        for obj in obj_array:
            topleft = obj.bounding_box_2d[0]
            bottomright = obj.bounding_box_2d[2]

            # Avstand til midten av objektet
            middle = (int(topleft[0] + (bottomright[0] - topleft[0]) / 2), int(topleft[1] + (bottomright[1] - topleft[1]) / 2))

            distance = calculateDistance(middle, depth_map)

            if selected_object == str(obj.id):
                # Draw red bounding box and label
                cv2.rectangle(img_cv, (int(topleft[0]), int(topleft[1])), (int(bottomright[0]), int(bottomright[1])), (0, 0, 255), 2)
                label = f"{obj.id} ({int(obj.confidence)}%) dist: {distance:.2f}m"
                cv2.putText(img_cv, label, (int(topleft[0]), int(topleft[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                # Draw a dot with a diameter of 3 pixels at the center of the object
                cv2.circle(img_cv, (middle[0], middle[1]), 3, (255, 0, 0), -1)

            else:
                # Draw green bounding box and label
                cv2.rectangle(img_cv, (int(topleft[0]), int(topleft[1])), (int(bottomright[0]), int(bottomright[1])), (0, 255, 0), 2)
                label = f"{obj.id} ({int(obj.confidence)}%) dist: {distance:.2f}m"
                cv2.putText(img_cv, label, (int(topleft[0]), int(topleft[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                # Draw a dot with a diameter of 3 pixels at the center of the object
                cv2.circle(img_cv, (middle[0], middle[1]), 3, (255, 0, 0), -1)


def main_loop(zed, obj_runtime_param):
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

            # Calculate FPS
            calculateFPS()

            cv2.imshow("Object detection with ZED", img_cv)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.disable_object_detection()
    zed.close()
    cv2.destroyAllWindows()



def main():
    zed, obj_runtime_param = init()
    main_loop(zed, obj_runtime_param)


if __name__ == "__main__":
    main()
