from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
  def __init__(self):
    # Initialize the DeepSort object tracker with specific parameters
    self.object_tracker = DeepSort(
        max_age=35,  # Maximum number of frames to keep a track alive without detections
        n_init=3,  # Minimum number of detections required to confirm a track
        nms_max_overlap=0.3,  # Non-maximum suppression threshold for overlapping boxes
        max_cosine_distance=0.8,  # Maximum cosine distance for re-identification
        nn_budget=None,  # Budget for the nearest neighbor search
        embedder="mobilenet",  # Embedder model used for feature extraction
        half=True,  # Use half-precision for faster computation
        bgr=True,  # Specify that input frames are in BGR format
        embedder_model_name=None,  # Custom embedder model name (if any)
    )

  def track(self, detections, frame):
    # Update the tracker with new detections and the current frame
    tracks = self.object_tracker.update_tracks(detections, frame=frame)

    tracking_ids = []  # List to store IDs of confirmed tracks
    boxes = []  # List to store bounding boxes of confirmed tracks
    for track in tracks:
      if not track.is_confirmed():  # Skip tracks that are not confirmed
        continue
      tracking_ids.append(track.track_id)  # Add the track ID to the list
      ltrb = track.to_ltrb()  # Get the bounding box in (left, top, right, bottom) format
      boxes.append(ltrb)  # Add the bounding box to the list

    return tracking_ids, boxes  # Return the tracking IDs and bounding boxes