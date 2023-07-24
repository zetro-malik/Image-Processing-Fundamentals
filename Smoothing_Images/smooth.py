# import required packages
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
# Read images for different blurring purposes
# Read the image in its original color space
image = cv2.imread(r"Smoothing_Images\test_img.jpg")

# Convert the image to RGB color space
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Blur images
image_MedianBlur = cv2.medianBlur(image, 9)
image_GaussianBlur = cv2.GaussianBlur(image, (9, 9), 10)
image_BilateralBlur = cv2.bilateralFilter(image, 9,
                                          100, 75)
# Display images
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

axs[0, 0].imshow(image)
axs[0, 0].set_title('ORGINAL')

axs[0, 1].imshow(image_MedianBlur)
axs[0, 1].set_title('image_MedianBlur')

axs[1, 0].imshow(image_GaussianBlur)
axs[1, 0].set_title('image_GaussianBlur')

axs[1, 1].imshow(image_BilateralBlur)
axs[1, 1].set_title('image_BilateralBlur')


plt.tight_layout()
plt.show()
