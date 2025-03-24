import pyzed.sl as sl
import numpy as np
import cv2

import os
import threading

## Global variables
active_objects = []
last_active_objects = []    
selected_object = None 





def init_zed():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1
    init_params.camera_resolution = sl.RESOLUTION.HD720

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


def init_CV_window():
    cv2.namedWindow("ZED", cv2.WINDOW_NORMAL)


# Samler alle initialiseringsfunksjonene i en funksjon 
def init():
    zed = init_zed()
    print("ZED initialized")
    _, obj_runtime_param = init_object_detection(zed)
    print("Object detection initialized")
    init_CV_window()
    print("Window initialized")
    kthread = KeyboardThread(input_cbk=my_callback)
    print("Keyboard thread initialized")

    return zed, obj_runtime_param


def calculateDistance(middle, depth_map, point_cloud, img):
                # Metode 1 for å finne avstand til objektet
    distanceMetod1 = False
    if distanceMetod1 == True:
        depth_value = depth_map.get_value(middle[0], middle[1])
        distance = depth_value[1]

            # Metode 2 for å finne avstand til objektet
    distanceMetod2 = False
    if distanceMetod2 == True:
        point3D = point_cloud.get_value(middle[0], middle[1])
        x, y, z = point3D[1][:3]
        distance = np.sqrt(x**2 + y**2 + z**2)

            # Metode 3 for å finne avstand til objektet
    distanceMetod3 = True
    if distanceMetod3 == True:
                #x = round(img.get_width() / 2)
                #y = round(img.get_height() / 2)
        x = middle[0]
        y = middle[1]
        err, point_cloud_value = point_cloud.get_value(x, y)
        distance = np.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                            point_cloud_value[1] * point_cloud_value[1] +
                            point_cloud_value[2] * point_cloud_value[2])
    return distance


def process_objects(objects, img_cv, depth_map, point_cloud, img):
    global active_objects
    active_objects = []
    global selected_object


    if objects.is_new:
        obj_array = objects.object_list
        #print(f"{len(obj_array)} Object(s) detected")

        for obj in obj_array:
            topleft = obj.bounding_box_2d[0]
            bottomright = obj.bounding_box_2d[2]
            V = obj.velocity
            V_tot = np.round(np.sqrt(V[0]**2 + V[1]**2 + V[2]**2) * 3.6, 3)

            # Add object to active objects list
            active_objects.append([obj.id, obj.label])

            # Avstand til midten av objektet
            middle = (int(topleft[0] + (bottomright[0] - topleft[0]) / 2), int(topleft[1] + (bottomright[1] - topleft[1]) / 2))

            distance = calculateDistance(middle, depth_map, point_cloud, img)

            if selected_object == str(obj.id):
                # Draw red bounding box and label
                cv2.rectangle(img_cv, (int(topleft[0]), int(topleft[1])), (int(bottomright[0]), int(bottomright[1])), (0, 0, 255), 2)
                label = f"{obj.id} ({int(obj.confidence)}% Velo: {V_tot} km/h) dist: {distance:.2f}m"
                cv2.putText(img_cv, label, (int(topleft[0]), int(topleft[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                # Draw a dot with a diameter of 3 pixels at the center of the object
                cv2.circle(img_cv, (middle[0], middle[1]), 3, (255, 0, 0), -1)

            else:
                # Draw green bounding box and label
                cv2.rectangle(img_cv, (int(topleft[0]), int(topleft[1])), (int(bottomright[0]), int(bottomright[1])), (0, 255, 0), 2)
                label = f"{obj.id} ({int(obj.confidence)}% Velo: {V_tot} km/h) dist: {distance:.2f}m"
                cv2.putText(img_cv, label, (int(topleft[0]), int(topleft[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                # Draw a dot with a diameter of 3 pixels at the center of the object
                cv2.circle(img_cv, (middle[0], middle[1]), 3, (255, 0, 0), -1)

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


def main_loop(zed, obj_runtime_param):
    objects = sl.Objects()
    
    print("Window initialized")
    
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            img = sl.Mat()
            zed.retrieve_image(img, sl.VIEW.LEFT)
            img_cv = np.array(img.get_data(), dtype=np.uint8)

            zed.retrieve_objects(objects, obj_runtime_param)

            #distanse metode 1
            depth_map = sl.Mat()
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            #distanse metode 2
            point_cloud = sl.Mat()
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            

            active_objects = process_objects(objects, img_cv, depth_map, point_cloud, img)
            print_active_objects(active_objects)

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
