from ultralytics import YOLO
import pyzed.sl as sl
import numpy as np
import cv2
import os


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
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the model file
    model_path = os.path.join(script_dir, "Models", "yolov8s.engine")
    model = YOLO(model_path, task="detect")
    print("YOLO model initialized")
    return model


def process_yolo_results(results, img_cv, depth_map, point_cloud):
    # Define the classes to keep
    names = {
        0: 'person',
        1: 'car',
        2: 'motorcycle',
        4: 'airplane',
        4: 'bus',
        5: 'train',
        6: 'truck',
        7: 'boat',
    }

    # Process YOLO results and draw bounding boxes on the image
    for r in results:
        for box in r.boxes:
            label_id = int(box.cls[0])  # Class ID
            if label_id in names:  # Filter by desired classes
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0]  # Confidence score
                label = names[label_id]  # Get class label from the dictionary
                distance = calculateDistance("depth", (x1, y1), (x2, y2), depth_map, point_cloud)

                # Draw bounding box
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img_cv, (int(x1-1), int(y1-23)), (int(x2+1), int(y1)), (0, 255, 0), -1)

                # Add label and confidence
                label = f"{label} Conf: {int(confidence*100)}%  Dist: {distance:.2f}m"
                cv2.putText(img_cv, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


def calculateDistance(Type, topleft, bottomright, depth_map, point_cloud):
    middle = (int(topleft[0] + (bottomright[0] - topleft[0]) / 2), int(topleft[1] + (bottomright[1] - topleft[1]) / 2))

    if Type == "depth":
        depth_value = depth_map.get_value(middle[0], middle[1])
        _, distance = depth_value
        return distance
    elif Type == "point_cloud":
        point3D = point_cloud.get_value(middle[0], middle[1])
        x, y, z = point3D[1][:3]
        distance = np.sqrt(x**2 + y**2 + z**2)
        return distance
       

def main_loop(zed, model):
    # Create a ZED Mat object to store images
 

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image from the ZED camera
            zed_image = sl.Mat()
            depth_map = sl.Mat()
            point_cloud = sl.Mat()

            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            
            img_cv = np.array(zed_image.get_data(), dtype=np.uint8)

            # Convert RGBA to RGB
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)

            # Predict using YOLO
            results = model.predict(img_cv, stream=True)

            # Process results and draw on the frame
            process_yolo_results(results, img_cv, depth_map, point_cloud)

            # Display the frame
            cv2.imshow("YOLO Object Detection with ZED", img_cv)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()


def main():
    # Initialize ZED camera
    zed = init_zed()

    # Initialize YOLO model
    model = init_yolo()

    # Start the main loop
    main_loop(zed, model)


if __name__ == "__main__":
    main()