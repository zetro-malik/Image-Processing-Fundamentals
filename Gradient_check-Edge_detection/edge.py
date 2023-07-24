"""
edges in an image correspond to areas of high gradient,
 where the image intensity changes rapidly from one pixel to the next. By detecting these edges, 
 we can extract important features from an image and use them for various tasks such as object detection and recognition, 
 image registration, and image enhancement.
"""


# Import packages
import cv2
# Read image
src = cv2.resize(cv2.imread(
    r"Gradient_check-Edge_detection\hand.jpg"), (640, 640))
# Apply gaussian blur
cv2.GaussianBlur(src, (3, 3), 0)
# Convert image to grayscale
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# Apply Sobel method to the grayscale image

# Getting edges of X-axis
grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1,
                   delta=0, borderType=cv2.BORDER_DEFAULT)  # Horizontal Sobel Derivation
# Getting edges of Y-axis
grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1,
                   delta=0, borderType=cv2.BORDER_DEFAULT)  # Vertical Sobel Derivation

# take absolute value of each pixel
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
grad = cv2.addWeighted(abs_grad_x, 0.9, abs_grad_y, 0.2, 0)
# Apply both
# Show the image
cv2.imshow("img", grad)  # View the image

cv2.waitKey(0)

cv2.destroyAllWindows()
