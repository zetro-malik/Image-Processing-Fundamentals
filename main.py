import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = (image - min_val) * (255.0 / (max_val - min_val))
    return stretched.astype(np.uint8)

def histogram_equalization(image):
    equalized = cv2.equalizeHist(image)
    return equalized

def adaptive_histogram_equalization(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    equalized = clahe.apply(image)
    return equalized

def sharpen(image):
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred

def median_blur(image, kernel_size=5):
    blurred = cv2.medianBlur(image, kernel_size)
    return blurred

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return filtered

def adaptive_thresholding(image, max_value=255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                          threshold_type=cv2.THRESH_BINARY, block_size=11, constant=2):
    thresholded = cv2.adaptiveThreshold(image, max_value, adaptive_method, threshold_type, block_size, constant)
    return thresholded



def saturation_adjustment(image, saturation_factor):
    adjusted = image * saturation_factor
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


def invert_image(image):
    inverted = 255 - image
    return inverted

def gamma_correction(image, gamma=1.0):
    gamma_corrected = np.power(image / 255.0, gamma) * 255.0
    return gamma_corrected.astype(np.uint8)

def edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    return rotated

# Load and preprocess the image
image_path = r'Image_Thresholding\hand.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply each enhancement function one by one
enhanced_image_1 = contrast_stretching(image)
enhanced_image_2 = histogram_equalization(image)
enhanced_image_3 = adaptive_histogram_equalization(image)
enhanced_image_4 = sharpen(image)
enhanced_image_5 = gaussian_blur(image)
enhanced_image_6 = median_blur(image)
enhanced_image_7 = bilateral_filter(image)
enhanced_image_8 = adaptive_thresholding(image)
enhanced_image_9 = saturation_adjustment(image, saturation_factor=1.5)
enhanced_image_10 = invert_image(image)
enhanced_image_11 = gamma_correction(image, gamma=1.5)
enhanced_image_12 = edge_detection(image)
enhanced_image_13 = rotate_image(image, angle=45)

# Display the enhanced images using Matplotlib
plt.figure(figsize=(15, 12))
plt.subplot(4, 4, 1), plt.imshow(enhanced_image_1, cmap='gray'), plt.title('Contrast Stretching')
plt.subplot(4, 4, 2), plt.imshow(enhanced_image_2, cmap='gray'), plt.title('Histogram Equalization')
plt.subplot(4, 4, 3), plt.imshow(enhanced_image_3, cmap='gray'), plt.title('Adaptive Histogram Equalization')
plt.subplot(4, 4, 4), plt.imshow(enhanced_image_4, cmap='gray'), plt.title('Sharpening')
plt.subplot(4, 4, 5), plt.imshow(enhanced_image_5, cmap='gray'), plt.title('Gaussian Blur')
plt.subplot(4, 4, 6), plt.imshow(enhanced_image_6, cmap='gray'), plt.title('Median Blur')
plt.subplot(4, 4, 7), plt.imshow(enhanced_image_7, cmap='gray'), plt.title('Bilateral Filtering')
plt.subplot(4, 4, 8), plt.imshow(enhanced_image_8, cmap='gray'), plt.title('Adaptive Thresholding')
plt.subplot(4, 4, 9), plt.imshow(enhanced_image_9, cmap='gray'), plt.title('Saturation Adjustment')
plt.subplot(4, 4, 10), plt.imshow(enhanced_image_10, cmap='gray'), plt.title('Invert Image')
plt.subplot(4, 4, 11), plt.imshow(enhanced_image_11, cmap='gray'), plt.title('Gamma Correction')
plt.subplot(4, 4, 12), plt.imshow(enhanced_image_12, cmap='gray'), plt.title('Edge Detection')
plt.subplot(4, 4, 13), plt.imshow(enhanced_image_13, cmap='gray'), plt.title('Rotate Image')

plt.tight_layout()
plt.show()
