import cv2
import numpy as np
import pyzed.sl as sl


zed = sl.Camera()

## Setter paprametre for cameraet (resulution, fps, etc.)
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

err = zed.open(init_params)


## Setter parametre for objekt detektion (tracking, segmentation, model, etc.)
obj_param = sl.ObjectDetectionParameters()
obj_param.enable_tracking = True
obj_param.enable_segmentation = True
#obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.PERSON_HEAD_DETECTION
obj_param.object_detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_DETECTION

if obj_param.enable_tracking:
    positional_tracking_param = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_param)


err = zed.enable_object_detection(obj_param)



objects = sl.Objects()
obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
obj_runtime_param.detection_confidence_threshold = 30



cv2.namedWindow("ZED", cv2.WINDOW_NORMAL)



while True:

    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_objects(objects, obj_runtime_param)

        img = sl.Mat()
        zed.retrieve_image(img, sl.VIEW.LEFT)

        img_cv = img.get_data()
        img_cv = np.array(img_cv, dtype=np.uint8)
        


        if objects.is_new:
            obj_arr = objects.object_list
            print(str(len(obj_arr)) + " objekter er detekteret")
            #print(obj_arr)

            for obj in obj_arr:
                print("Objekt: ")
                print(obj.bounding_box_2d[0])
                print(obj.bounding_box_2d[2])
                topleft = obj.bounding_box_2d[0]
                bottomright = obj.bounding_box_2d[2]

                cv2.rectangle(img_cv, (int(topleft[0]), int(topleft[1])), (int(bottomright[0]), int(bottomright[1])), (0, 255, 0), 2)


                label = f"{obj.label} ({int(obj.confidence)}%)"

                cv2.putText(img_cv, label, (int(topleft[0]), int(topleft[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


        cv2.imshow("Object detection with ZED", img_cv)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


zed.disable_object_detection()
zed.close()
cv2.destroyAllWindows()

                




