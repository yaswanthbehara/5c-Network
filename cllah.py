import cv2
import os

def apply_clahe(image):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def preprocess_images(data_path):

    processed_images = []
    for file in os.listdir(data_path):
        if not file.endswith("_mask.tif"):  # Ignore mask files
            image = cv2.imread(os.path.join(data_path, file), cv2.IMREAD_GRAYSCALE)
            image = apply_clahe(image)
            processed_images.append(image)
    return processed_images
