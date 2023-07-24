import cv2
import numpy as np

# Load image
image_path = "test_img.jpg"
img = cv2.imread(image_path)

# Gamma correction parameter
gamma = 0.4

# Apply gamma correction
gamma_corrected = np.power(img / 255.0, gamma)
gamma_corrected = np.uint8(gamma_corrected * 255)

# Display original and gamma corrected image
cv2.imshow("Original", cv2.resize(img, (640, 640)))
cv2.imshow("Gamma Corrected", cv2.resize(gamma_corrected, (640, 640)))
cv2.waitKey(0)
cv2.destroyAllWindows()
