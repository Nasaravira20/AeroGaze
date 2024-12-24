import cv2
import numpy as np

img_path = ''
img = cv2.imread(img_path)

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Could not load image at {img_path}. Check the file path or file permissions.")
    exit()

# Convert image from BGR to HSV for better color manipulation
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the range of pink color in HSV space and create a mask
lower_pink = np.array([140, 50, 50])   # Lower bound for pink
upper_pink = np.array([180, 255, 255]) # Upper bound for pink
mask = cv2.inRange(hsv, lower_pink, upper_pink)

# Reduce the pink tint by decreasing saturation and increasing brightness in masked areas
hsv[:, :, 1] = cv2.subtract(hsv[:, :, 1], mask)  # Decrease saturation
hsv[:, :, 2] = cv2.add(hsv[:, :, 2], mask // 2)  # Increase brightness slightly

# Convert back to BGR
result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Save the result
output_path = '/home/cyberkid/Desktop/raw_model/normalized_image.png'
cv2.imwrite(output_path, result)

print(f"Image has been converted and saved as {output_path}.")
