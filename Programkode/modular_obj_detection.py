import pyzed.sl as sl
import numpy as np
import cv2


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

    return zed, obj_runtime_param



def process_objects(objects, img_cv):
    if objects.is_new:
        obj_array = objects.object_list
        #print(f"{len(obj_array)} Object(s) detected")

        for obj in obj_array:
            topleft = obj.bounding_box_2d[0]
            bottomright = obj.bounding_box_2d[2]

            cv2.rectangle(img_cv, (int(topleft[0]), int(topleft[1])), (int(bottomright[0]), int(bottomright[1])), (0, 255, 0), 2)

            V = obj.velocity
            V_tot = np.round(np.sqrt(V[0]**2 + V[1]**2 + V[2]**2) * 3.6, 3)

            label = f"{obj.label} ({int(obj.confidence)}% Velo: {V_tot} km/h)"
            cv2.putText(img_cv, label, (int(topleft[0]), int(topleft[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)



def main_loop(zed, obj_runtime_param):
    objects = sl.Objects()
    
    
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            img = sl.Mat()
            zed.retrieve_image(img, sl.VIEW.LEFT)
            img_cv = np.array(img.get_data(), dtype=np.uint8)

            zed.retrieve_objects(objects, obj_runtime_param)
            process_objects(objects, img_cv)

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
