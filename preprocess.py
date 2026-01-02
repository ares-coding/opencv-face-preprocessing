"""
OpenCV Face Preprocessing Script
Author: Ares Coding

This script demonstrates basic face image preprocessing steps
commonly used in Machine Learning and Computer Vision projects.

Steps included:
1. Load images from a folder
2. Detect faces using Haar Cascade
3. Convert to grayscale
4. Resize images
5. Normalize pixel values
6. Save processed images
"""

import os
import cv2
import numpy as np

# ----------------------------
# Configuration
# ----------------------------

RAW_DIR = "dataset/raw_images"
PROCESSED_DIR = "dataset/processed_images"
IMAGE_SIZE = (128, 128)

# Create output directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----------------------------
# Preprocessing Function
# ----------------------------

def preprocess_image(image_path, output_path):
    """
    Preprocess a single image:
    - Read image
    - Detect face
    - Convert to grayscale
    - Resize
    - Normalize
    - Save processed image
    """

    image = cv2.imread(image_path)

    if image is None:
        print(f"[WARNING] Could not read image: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        print(f"[INFO] No face detected: {image_path}")
        return

    # Use the first detected face
    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]

    # Resize face
    face_resized = cv2.resize(face, IMAGE_SIZE)

    # Normalize pixel values (0–1)
    face_normalized = face_resized / 255.0

    # Convert back to uint8 for saving
    face_uint8 = (face_normalized * 255).astype("uint8")

    cv2.imwrite(output_path, face_uint8)
    print(f"[OK] Processed: {output_path}")

# ----------------------------
# Main Loop
# ----------------------------

def main():
    for filename in os.listdir(RAW_DIR):
        input_path = os.path.join(RAW_DIR, filename)
        output_path = os.path.join(PROCESSED_DIR, filename)

        preprocess_image(input_path, output_path)

    print("\nPreprocessing complete.")

if __name__ == "__main__":
    main()
