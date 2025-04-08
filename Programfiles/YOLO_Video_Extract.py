
import pyzed.sl as sl
import cv2
import numpy as np
 
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

def save_video_from_zed(output_file="output_video.mp4", fps=30):
    # Initialize ZED camera
    zed = init_zed()

    # Create a ZED Mat object to store images
    zed_image = sl.Mat()

    # Get the resolution of the ZED camera
    resolution = zed.get_camera_information().camera_resolution
    width = resolution.width
    height = resolution.height

    # Initialize OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Codec for MP4 files
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print(f"Recording video to {output_file}... Press 'q' to stop.")

    while True:
        # Grab a new frame from the ZED camera
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image from the ZED camera
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            frame = np.array(zed_image.get_data(), dtype=np.uint8)

            # Convert RGBA to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            # Write the frame to the video file
            out.write(frame)

            # Display the frame
            cv2.imshow("ZED Camera", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 
# Release resources
    out.release()
    zed.close()
    cv2.destroyAllWindows()
    print(f"Video saved to {output_file}")

if __name__ == "__main__":
    save_video_from_zed(output_file="zed_output.mp4", fps=30)