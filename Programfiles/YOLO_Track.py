from ultralytics import YOLO
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
    model_path = os.path.join(script_dir, "Models/yolov8s.engine")
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
    print('You selected object:', inp)
    global selected_object
    selected_object = inp

def process_yolo_results(results, img_cv):

    global active_objects
    global selected_object

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
                type = names[label_id]  # Get class label from the dictionary
                ID = int(box.id[0])  # Get the unique ID of the object

                active_objects.append([ID, type]) # Add object to the list of active objects

                if selected_object == str(ID):
                    # Draw bounding box
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # Add label and confidence
                    label_text = f"{ID} {type} ({confidence:.2f})"  # Updated to show object name
                    cv2.putText(img_cv, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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


def main_loop(zed, model):
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
            results = model.track(img_cv, stream=True, augment=True, verbose=False, device=0)

            # Process results and draw on the frame
            active_objects = process_yolo_results(results, img_cv)

            print_active_objects(active_objects)

            # Resize the frame to fixed dimensions
            resized_frame = cv2.resize(img_cv, (fixed_width, fixed_height))

            # Display the resized frame
            cv2.imshow("YOLO Object Detection with ZED", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()


def main():
    # Initialize ZED camera
    zed = init_zed()

    # Initialize YOLO model
    model = init_yolo()

    # Start the keyboard input thread
    kthread = KeyboardThread(my_callback)

    # Start the main loop
    main_loop(zed, model)


if __name__ == "__main__":
    main()
