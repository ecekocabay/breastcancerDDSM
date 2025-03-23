# Step 6: Compute LBP Features
def compute_lbp(image, radius=1, n_points=8):
    if image.max() > 1:
        image = (image / image.max() * 255).astype(np.uint8)
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)
    return hist

# Process Image
def process_image(filepath, radius=1, n_points=8):
    original_image = load_dicom_as_image(filepath)
    if original_image is None:
        return None

    contrast_image = contrast_enhancement(original_image)
    seed_point = (contrast_image.shape[1] // 2, contrast_image.shape[0] // 2)
    region_grown = region_growing(contrast_image, seed_point, threshold=15)
    refined_image = morphological_operations(region_grown)
    contour_image, contours = contour_extraction(refined_image)
    cropped_image = crop_image_with_contours(original_image, contours)

    lbp_hist = compute_lbp(cropped_image, radius=radius, n_points=n_points)
    return lbp_hist
