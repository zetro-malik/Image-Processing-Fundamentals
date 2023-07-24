# import required packages
import cv2
from skimage import io
# Read image 1
img1 = cv2.resize(cv2.imread('Blending_imgs_openCV\puppy.jpg'), (640, 640))
# Read image 2
img2 = cv2.resize(cv2.imread('Blending_imgs_openCV\puppy2.jpg'), (640, 640))
# Define alpha and beta
alpha = 0.80
beta = 0.50
# Blend images
final_image = cv2.addWeighted(img1, alpha, img2, beta, 0.0)
# Show image
cv2.imshow("final blended", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(r"Blending_imgs_openCV\blended.jpg", final_image)
