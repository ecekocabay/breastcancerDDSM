import os
import numpy as np
import pandas as pd
import cv2
import pydicom
from collections import deque
from skimage.feature import local_binary_pattern

# Step 0: Load DICOM file
def load_dicom_as_image(dicom_path):
    try:
        dicom = pydicom.dcmread(dicom_path, force=True)
        pixel_array = dicom.pixel_array.astype(np.float32)
        pixel_array -= pixel_array.min()
        pixel_array /= (pixel_array.max() + 1e-6)
        pixel_array *= 255.0
        return pixel_array.astype(np.uint8)
    except Exception as e:
        print(f":x: Error reading {dicom_path}: {e}")
        return None

# Step 1: Contrast Enhancement using CLAHE and Median Filter
def contrast_enhancement(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    enhanced = cv2.medianBlur(enhanced, 5)
    return enhanced

# Step 2: Morphological Operations to Refine Segmentation
def morphological_operations(binary_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (1000, 1000))
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return cleaned_image

# Step 3: Region Growing Algorithm
def region_growing(image, seed_point, threshold=8):
    h, w = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    stack = deque([seed_point])
    seed_intensity = image[seed_point[1], seed_point[0]]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while stack:
        x, y = stack.pop()
        if segmented[y, x] == 0:
            segmented[y, x] = 255
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if segmented[ny, nx] == 0 and abs(int(image[ny, nx]) - int(seed_intensity)) <= threshold:
                        stack.append((nx, ny))
    return segmented

# Step 4: Contour Extraction
def contour_extraction(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(binary_image)
    cv2.drawContours(contour_image, contours, -1, 255, 2)
    return contour_image, contours

# Step 5: Crop Image
def crop_image_with_contours(original_image, contours):
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_region = original_image[y:y+h, x:x+w]
        return cropped_region
    return original_image