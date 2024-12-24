import cv2
from bytetrack import BYTETracker

# Initialize the tracker
tracker = BYTETracker()

# Start the video capture (or frame processing)
cap = cv2.VideoCapture('/home/cyberkid/Desktop/jaya.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Here, you would use a pre-trained object detector to get detections
    # Example: detections is a list of [xmin, ymin, xmax, ymax, confidence, class_id]
    detections = []  # Replace with your actual detections, e.g., YOLO detections

    # Update the tracker with new detections and get tracking results
    track_results = tracker.update(detections)

    # Now, count the number of objects being tracked
    tracked_objects = len(track_results)

    print(f"Tracked objects: {tracked_objects}")

    # (Optional) Draw tracking results on the frame for visualization
    for track in track_results:
        # track is [track_id, x1, y1, x2, y2, etc.]
        cv2.rectangle(frame, (int(track[1]), int(track[2])), (int(track[3]), int(track[4])), (0, 255, 0), 2)
        cv2.putText(frame, str(track[0]), (int(track[1]), int(track[2])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
