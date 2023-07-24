# import required packages
import cv2
import numpy as np
from skimage import io
# Read image
image = cv2.resize(cv2.imread("hand_img.jpg"), (640, 640))
# Define font
font = cv2.FONT_HERSHEY_SIMPLEX
# Write on the image
cv2.putText(image, "I am a Cat", (230, 50), font, 0.8, (0, 255, 0),
            2, cv2.LINE_AA)
cv2.imshow("image", image)
cv2.waitKey(0)

cv2.destroyAllWindows()
