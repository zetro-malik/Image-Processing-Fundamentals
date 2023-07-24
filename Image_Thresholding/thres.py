# Import packages
import cv2
from matplotlib import pyplot as plt
# Read image
image = cv2.imread(r"Image_Thresholding\hand.jpg")
# Define threshold types
"""
0 - Binary
1 - Binary Inverted
2 - Truncated
3 - Threshold To Zero
4 - Threshold To Zero Inverted
"""
# Apply different thresholds and save in different variables
_, img1 = cv2.threshold(image, 50, 255, 0)
_, img2 = cv2.threshold(image, 50, 255, 1)
_, img3 = cv2.threshold(image, 50, 255, 2)
_, img4 = cv2.threshold(image, 50, 255, 3)
_, img5 = cv2.threshold(image, 50, 255, 4)
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))

# Show the different threshold images
axs[0, 0].imshow(img1)
axs[0, 0].set_title('Binary')

axs[0, 1].imshow(img2)
axs[0, 1].set_title('Binary Inverted')

axs[0, 2].imshow(img3)
axs[0, 2].set_title('Truncated')

axs[1, 0].imshow(img4)
axs[1, 0].set_title('Threshold To Zero')

axs[1, 1].imshow(img5)
axs[1, 1].set_title('Threshold To Zero Inverted')

axs[1, 2].imshow(image)
axs[1, 2].set_title('ORIGINAL')


plt.tight_layout()
plt.show()
