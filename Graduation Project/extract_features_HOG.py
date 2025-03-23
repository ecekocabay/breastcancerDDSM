import pandas as pd
import numpy as np
from skimage.feature import hog
from skimage import io
import os

def compute_hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    """
    Computes the HOG (Histogram of Oriented Gradients) features for a given image.

    :param image: Grayscale image array.
    :param pixels_per_cell: Size of the cell (in pixels) for HOG computation.
    :param cells_per_block: Number of cells per block for normalization.
    :param orientations: Number of orientation bins for the histogram.
    :return: Flattened HOG feature vector.
    """
    if image.max() > 1:  # Normalization
        image = (image / image.max()).astype(np.float32)

    #  HOG features computing
    hog_features = hog(image, 
                       orientations=orientations, 
                       pixels_per_cell=pixels_per_cell, 
                       cells_per_block=cells_per_block, 
                       block_norm='L2-Hys', 
                       visualize=False, 
                       feature_vector=True)
    return hog_features

def extract_hog_features(image_folder, output_csv, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    """
    Extracts HOG features from all images in the specified folder and saves them to a CSV file.

    :param image_folder: Path to the folder containing images.
    :param output_csv: Path to save the extracted feature matrix as a CSV file.
    :param pixels_per_cell: Size of the cell (in pixels) for HOG computation.
    :param cells_per_block: Number of cells per block for normalization.
    :param orientations: Number of orientation bins for the histogram.
    """
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    feature_matrix = []
    image_names = []

    for image_file in image_files:
        image = io.imread(os.path.join(image_folder, image_file), as_gray=True)
        hog_features = compute_hog(image, 
                                   pixels_per_cell=pixels_per_cell, 
                                   cells_per_block=cells_per_block, 
                                   orientations=orientations)
        feature_matrix.append(hog_features)
        image_names.append(image_file)

    # creataring a DataFrame
    columns = [f"Feature_{i+1}" for i in range(len(feature_matrix[0]))]
    feature_matrix_df = pd.DataFrame(feature_matrix, columns=columns)
    feature_matrix_df.insert(0, "Image Name", image_names)

    # save to CSV
    feature_matrix_df.to_csv(output_csv, index=False)
    print(f"Feature matrix successfully saved to '{output_csv}'.")

if __name__ == "__main__":
    image_folder = r'/Users/ecekocabay/Desktop/BreastCancerDetection_noyan/output_images'  
    output_csv = '/Users/ecekocabay/Desktop/BreastCancerDetection_noyan/data/hog_feature_matrix.csv'  
    extract_hog_features(image_folder, output_csv)
