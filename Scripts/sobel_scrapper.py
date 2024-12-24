import os
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from collections import defaultdict

# Directories and model paths
input_folder = '/home/cyberkid/Desktop/raw_model/Result/'
output_folder = 'cropped_image'
prediction_output = 'prediction.txt'
model_path_scrapper = "/home/cyberkid/Desktop/raw_model/scrapper.pt"
model_path_shape_counter = "/home/cyberkid/Desktop/raw_model/detection_counter.pt"
cwd = os.getcwd()

# Clear the output folder and create it if it doesn't exist
os.system(f'rm -f {output_folder}/*')
os.makedirs(output_folder, exist_ok=True)

# Set the device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the models
model_scrapper = YOLO(model_path_scrapper)
model_shape_counter = YOLO(model_path_shape_counter)

# Preprocess function
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude)) if np.max(gradient_magnitude) > 0 else gradient_magnitude
    return gradient_magnitude

# Dictionary to store shape counts
shape_counts = defaultdict(int)

# Process each image in the folder
for image in os.listdir(input_folder):
    path = os.path.join(cwd, input_folder, image)
    img = cv2.imread(path)
    
    # Run YOLO model on the image for initial object detection
    results = model_scrapper(path)
    boxes = results[0].boxes

    if boxes is None:
        print(f"No object detected in {image}")
        continue

    for idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box.xyxy[0].int().tolist()
        conf = box.conf[0].item()
        cls = box.cls[0].item()

        if conf < 0.6:
            continue

        xmin, xmax = max(0, xmin), min(img.shape[1], xmax)
        ymin, ymax = max(0, ymin), min(img.shape[0], ymax)

        # Crop and preprocess the detected region
        cropped = img[ymin:ymax, xmin:xmax]
        cropped = preprocess_image(cropped)
        
        # Save the cropped image for verification
        file_name, _ = os.path.splitext(image)
        cropped_path = os.path.join(output_folder, f"{file_name}_{idx}_{conf:.2f}.jpg")
        cv2.imwrite(cropped_path, cropped)

        # Run shape_counter model on the cropped image
        shape_result = model_shape_counter(cropped_path)
        
        # Extract the highest confidence class label from shape_result
        if shape_result[0].boxes:
            shape_box = shape_result[0].boxes[0]
            shape_conf = shape_box.conf[0].item()
            shape_cls = int(shape_box.cls[0].item())
            
            # Define class names (adjust these to match the classes in shape_counter.pt)
            class_names = {0: 'triangle', 1: 'circle', 2: 'square'}
            shape_name = class_names.get(shape_cls, "unknown")
            
            # Increment the count for the detected shape
            shape_counts[shape_name] += 1
            print(f"Detected {shape_name} with confidence {shape_conf:.2f} in {file_name}")

# Write the shape counts to prediction.txt
with open(prediction_output, 'w') as f:
    for shape, count in shape_counts.items():
        f.write(f"{shape}: {count}\n")

print(f"Shape counts saved to {prediction_output}")
