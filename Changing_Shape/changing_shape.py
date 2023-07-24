# Import package
import cv2
from matplotlib import pyplot as plt

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))
# Read image
image = cv2.imread(r"Changing_Shape\puppy.jpg")

# Define erosion size
s1 = 0
s2 = 10
s3 = 10
# Define erosion type
t1 = cv2.MORPH_RECT
t2 = cv2.MORPH_CROSS
t3 = cv2.MORPH_ELLIPSE
# Define and save the erosion template
tmp1 = cv2.getStructuringElement(t1, (2*s1 + 1, 2*s1+1), (s1, s1))
tmp2 = cv2.getStructuringElement(t2, (2*s2 + 1, 2*s2+1), (s2, s2))
tmp3 = cv2.getStructuringElement(t3, (2*s3 + 1, 2*s3+1), (s3, s3))
# Apply the erosion template to the image and save in different variables
final1 = cv2.erode(image, tmp1)
final2 = cv2.erode(image, tmp2)
final3 = cv2.erode(image, tmp3)
# Show all the images with different erosions


axs[0, 0].imshow(final1)
axs[0, 0].set_title('MORPH_RECT erode')

axs[0, 1].imshow(final2)
axs[0, 1].set_title('MORPH_CROSS erode')

axs[0, 2].imshow(final3)
axs[0, 2].set_title('MORPH_ELLIPSE erode')


# EROSION CODE:
# Import packages
# Read images
image = cv2.imread(r"Changing_Shape\puppy.jpg")

# Define dilation size
d1 = 0
d2 = 10
d3 = 20
# Define dilation type
t1 = cv2.MORPH_RECT
t2 = cv2.MORPH_CROSS
t3 = cv2.MORPH_ELLIPSE
# Store the dilation templates
tmp1 = cv2.getStructuringElement(t1, (2*d1 + 1, 2*d1+1), (d1, d1))
tmp2 = cv2.getStructuringElement(t2, (2*d2 + 1, 2*d2+1), (d2, d2))
tmp3 = cv2.getStructuringElement(t3, (2*d3 + 1, 2*d3+1), (d3, d3))
# Apply dilation to the images
final1 = cv2.dilate(image, tmp1)
final2 = cv2.dilate(image, tmp2)
final3 = cv2.dilate(image, tmp3)

# Show the images
axs[1, 0].imshow(final1)
axs[1, 0].set_title('MORPH_RECT dialate')

axs[1, 1].imshow(final2)
axs[1, 1].set_title('MORPH_CROSS dialate')

axs[1, 2].imshow(final3)
axs[1, 2].set_title('MORPH_ELLIPSE dialate')


plt.tight_layout()
plt.show()
