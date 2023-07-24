
# import required packages
from matplotlib import figure
from skimage import io
import cv2
import numpy as np
# Read image
image = cv2.resize(cv2.imread(
    r"changing_contrast_brightness\hand_img.jpg"), (256, 256))
# Create a dummy image that stores different contrast and brightness
new_image = np.zeros(image.shape, image.dtype)
# Brightness and contrast parameters
contrast = 0.8
bright = 2
# Change the contrast and brightness
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        for c in range(image.shape[2]):
            new_image[y, x, c] = np.clip(
                contrast*image[y, x, c] + bright, 0, 255)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, "Original Image", (50, 50),
            font, 0.5, (0, 255, 0), 1, cv2.LINE_4)

cv2.imshow('img', image)
cv2.imshow('new img', new_image)

cv2.waitKey()
cv2.destroyAllWindows()
