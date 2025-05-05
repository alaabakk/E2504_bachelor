from ultralytics import YOLO
import numpy as np
import cv2
import os
import threading
import time

## Global variables
selected_object = 'q'

def init_webcam():
    # Open the default webcam (device 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    return cap

def init_yolo():
    # Initialize the YOLO model
    print("Initializing YOLO model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the model file
    model_path = os.path.join(script_dir, "../../Models/yolov8n.pt")
    model = YOLO(model_path, task="detect")
    print("YOLO model initialized")
    return model

def startup_message():
    print("\nYOLO Object Detection with Webcam")
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

        # Draw rectangle and overlay FPS text
        cv2.rectangle(img, (x - padding, y - h - padding), (x + w + padding, y + padding), bg_color, -1)
        cv2.putText(img, text, (x, y), font, scale, text_color, thickness)

class KeyboardThread(threading.Thread):
    def __init__(self, input_cbk=None, name='keyboard-input-thread'):
        self.input_cbk = input_cbk
        super(KeyboardThread, self).__init__(name=name, daemon=True)
        self.start()

    def run(self):
        while True:
            self.input_cbk(input())  # Waits to get input + Return

def my_callback(inp):
    # Evaluate the keyboard input
    print('You selected object:', inp)
    global selected_object
    selected_object = inp


def process_yolo_results(results, img_cv):
    global selected_object

    # Define the classes to keep
    names = {
        0: 'person',
    }

    # Convert the generator to a list
    results_list = list(results)
    if not results_list:
        print("No objects detected in the frame.")
        return

    # Process YOLO results and draw bounding boxes on the image
    for r in results_list:
        for box in r.boxes:
            label_id = int(box.cls[0])  # Class ID
            if label_id in names:  # Filter by desired classes
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0]  # Confidence score
                type = names[label_id]  # Get class label from the dictionary
                
                # Check if the box has an ID
                if box.id is not None:
                    ID = int(box.id[0])  # Get the unique ID of the object
                else:
                    ID = -1  # Assign a default ID if none is available

                if selected_object == str(ID):
                    # Draw bounding box with red color for selected object
                    draw_bounding_box(img_cv, x1, y1, x2, y2, (0, 0, 255), ID, type, confidence)

                else:
                    # Draw bounding box with green color for other objects
                    draw_bounding_box(img_cv, x1, y1, x2, y2, (0, 255, 0), ID, type, confidence)

def draw_bounding_box(img_cv, x1, y1, x2, y2, color, tracking_id, type, confidence):
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
    label_text = f"{tracking_id} {type} ({confidence:.2f})"
    cv2.putText(img_cv, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main_loop_webcam(cap, model, fps_counter, fps):
    # Define fixed width and height
    fixed_width = 1280
    fixed_height = 720

    while True:
        ret, img_cv = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        try:
            # Predict using YOLO
            results = model.track(img_cv, stream=True, conf=0.5, tracker="custom_botsort.yaml", persist=True)

            # Process results and draw on the frame
            if results is not None:
                process_yolo_results(results, img_cv)
            else:
                print("No results from YOLO model.")

        except IndexError as e:
            print(f"IndexError encountered: {e}. Skipping this frame.")
            continue

        # Calculate and draw FPS
        fps_new, updated = fps_counter.calculateFPS()
        if updated:
            fps = fps_new
        fps_counter.draw_fps(img_cv, fps)

        # Resize the frame to fixed dimensions
        resized_frame = cv2.resize(img_cv, (fixed_width, fixed_height))
        cv2.imshow("YOLO Object Detection with Webcam", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    # Initialize webcam
    cap = init_webcam()

    # Initialize YOLO model
    model = init_yolo()

    # Start the keyboard input thread
    kthread = KeyboardThread(my_callback)

    # Initialize FPS counter
    fps_counter = FPSCounter()
    fps = 0

    startup_message()

    # Start the main loop for webcam
    main_loop_webcam(cap, model, fps_counter, fps)

if __name__ == "__main__":
    main()