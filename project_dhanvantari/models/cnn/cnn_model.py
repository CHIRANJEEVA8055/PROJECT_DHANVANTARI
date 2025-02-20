import os
import cv2
import numpy as np
from tqdm import tqdm

# Define paths
BASE_PATH = "C:/Users/INDUSTRY 4.0/PycharmProjects/project_dhanvantari/data/cnn"
RAW_DATA_PATH = os.path.join(BASE_PATH, "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_PATH, "processed")

# Create processed data directory if it doesn't exist
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)


def preprocess_image(image_path, target_size=(224, 224)):
    """Loads, resizes, and normalizes an image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read {image_path}")
        return None
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1] range
    return image


def process_images():
    """Processes all images in the raw data folder."""
    image_files = [f for f in os.listdir(RAW_DATA_PATH) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        print("No valid image files found in raw data directory.")
        return

    for filename in tqdm(image_files, desc="Processing Images"):
        file_path = os.path.join(RAW_DATA_PATH, filename)
        processed_image = preprocess_image(file_path)
        if processed_image is not None:
            save_path = os.path.join(PROCESSED_DATA_PATH, os.path.splitext(filename)[0] + ".npy")
            np.save(save_path, processed_image)
    print("Image preprocessing complete!")


if __name__ == "__main__":
    process_images()
