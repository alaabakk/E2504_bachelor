import pyzed.sl as sl
import cv2
import numpy as np

def main():
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


    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    cv2.namedWindow("ZED", cv2.WINDOW_NORMAL)


    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            img = sl.Mat()
            zed.retrieve_image(img, sl.VIEW.LEFT)
            img_cv = img.get_data()
            img_cv = np.array(img_cv, dtype=np.uint8)


            

            zed.retrieve_objects(objects, obj_runtime_param)

            if objects.is_new :
                obj_array = objects.object_list
                print(str(len(obj_array))+" Object(s) detected\n")

                if len(obj_array) > 0 :
                    first_object = obj_array[0]
                    print("First object attributes:")
                    print(" Label '"+repr(first_object.label)+"' (conf. "+str(int(first_object.confidence))+"/100)")

                    if obj_param.enable_tracking :
                        print(" Tracking ID: "+str(int(first_object.id))+" tracking state: "+repr(first_object.tracking_state)+" / "+repr(first_object.action_state))
                    position = first_object.position
                    velocity = first_object.velocity
                    dimensions = first_object.dimensions
                    print(" 3D position: [{0},{1},{2}]\n Velocity: [{3},{4},{5}]\n 3D dimentions: [{6},{7},{8}]".format(position[0],position[1],position[2],velocity[0],velocity[1],velocity[2],dimensions[0],dimensions[1],dimensions[2]))
                    
                    if first_object.mask.is_init():
                        print(" 2D mask available")

                    print(" Bounding Box 2D ")
                    bounding_box_2d = first_object.bounding_box_2d

                    for it in bounding_box_2d :
                        print("    "+str(it),end='')
                    print("\n Bounding Box 3D ")
                    bounding_box = first_object.bounding_box
                    
                    for it in bounding_box :
                        print("    "+str(it),end='')

                n = 0
                for obj in obj_array:
                    
                    print("Objekt: ")
                    print(obj.bounding_box_2d[0])
                    print(obj.bounding_box_2d[2])
                    topleft = obj.bounding_box_2d[0]
                    bottomright = obj.bounding_box_2d[2]

                    cv2.rectangle(img_cv, (int(topleft[0]), int(topleft[1])), (int(bottomright[0]), int(bottomright[1])), (0, 255, 0), 2)

                    V = obj_array[n].velocity
                    V_tot = np.round(np.sqrt(V[0]**2 + V[1]**2 + V[2]**2)*3.6, 3)

                    label = f"{obj.label} ({int(obj.confidence)}% Velo: {V_tot} km/h)"
                    

                    cv2.putText(img_cv, label, (int(topleft[0]), int(topleft[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) 

                    n += 1     


            cv2.imshow("Object detection with ZED", img_cv)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    # Close the camera
    zed.disable_object_detection()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()