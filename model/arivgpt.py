import os
from ultralytics import YOLO
import torch
import cv2
import numpy as np

folder = '/home/asunama/Desktop/last_model/'
# Make sure you have changed the directory which contains all the images of that dataset

output_folder = '/home/asunama/Desktop/data'
# model_path = "/home/cyberkid/Desktop/dataset/final.pt"
model_path = "/home/asunama/Desktop/best.pt"
cwd = os.getcwd()

os.system(f'rm {output_folder}/*')
os.makedirs(output_folder, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO(model_path)  

def preprocess_image(image):
    # Convert to grayscale if the image is in color
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to the grayscale image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Compute Sobel gradients in the x and y directions
    sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the gradient magnitude (take absolute values to avoid negative gradients)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize the gradient magnitude image to the range [0, 255]
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude)) if np.max(gradient_magnitude) > 0 else gradient_magnitude

    # Debugging: Display images at different stages
    cv2.imshow('Original Image', image)
    cv2.imshow('Gray Image', gray_image)
    cv2.imshow('Blurred Image', blurred_image)
    cv2.imshow('Sobel X', sobelx)
    cv2.imshow('Sobel Y', sobely)
    cv2.imshow('Gradient Magnitude', gradient_magnitude)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return gradient_magnitude

for image in os.listdir(folder):
    path = os.path.join(cwd, folder, image)
    img = cv2.imread(path)
    
    results = model(path)  

    boxes = results[0].boxes  
    if boxes is None:
        print('No object detected')
        continue
    for idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box.xyxy[0].int().tolist() 
        conf = box.conf[0].item() 
        cls = box.cls[0].item() 


        if conf < 0.3:
            continue
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

        xmin, xmax = max(0, xmin), min(img.shape[1], xmax)
        ymin, ymax = max(0, ymin), min(img.shape[0], ymax)

        cropped = img[ymin:ymax, xmin:xmax]

        # Preprocess the cropped image
        cropped = preprocess_image(cropped)

        file_name, file_extension = os.path.splitext(image)
        opath = os.path.join(output_folder, f"{file_name}{idx}{conf}.jpg")
        cv2.imwrite(opath, cropped)
        print(f"Detection saved as {opath}")
