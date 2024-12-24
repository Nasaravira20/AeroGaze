import os

import cv2
import numpy as np


def apply_sobel_filter(input_folder, output_folder="Result"):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over each file in the input folder
    for filename in os.listdir(input_folder):
        # Construct full file path
        img_path = os.path.join(input_folder, filename)

        # Check if it is a file and if it has a valid image extension
        if os.path.isfile(img_path) and filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        ):
            # Load the image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Apply the Sobel filter in the x and y directions
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

            # Calculate the gradient magnitude
            sobel_combined = cv2.magnitude(sobel_x, sobel_y)

            # Convert back to 8-bit image format
            sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

            # Save the processed image to the output folder
            output_path = os.path.join(output_folder, f"Sobel_{filename}")
            cv2.imwrite(output_path, sobel_combined)
            print(f"Processed and saved: {output_path}")


# Example usage:
# Replace 'path_to_images_folder' with the actual path to the folder containing your images
apply_sobel_filter("/home/cyberkid/Desktop/raw_model/Result")
