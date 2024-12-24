from ultralytics import YOLO
import cv2
from collections import defaultdict

def count_objects_in_video(video_path, model_path, output_txt, confidence_threshold=0.5):
    # Load the YOLO model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    frame_count = 0
    total_object_counts = defaultdict(int)  # To store object counts across all frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Perform inference on the current frame
        results = model(frame, conf=confidence_threshold)

        # Count objects in the current frame
        frame_object_counts = defaultdict(int)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])  # Class ID
                class_name = model.names[cls]  # Class name
                frame_object_counts[class_name] += 1

        # Aggregate counts across all frames
        for obj, count in frame_object_counts.items():
            total_object_counts[obj] += count

    cap.release()

    # Write total counts to the output file
    with open(output_txt, 'w') as f:
        for obj, count in total_object_counts.items():
            f.write(f"{obj}: {count}\n")
    
    print(f"Processing complete. Results saved to {output_txt}")

# Paths
video_file = "/home/cyberkid/desktop/raw.mp4"  # Replace with your input video file
model_file = "/home/cyberkid/Desktop/best.pt"  # Rep
