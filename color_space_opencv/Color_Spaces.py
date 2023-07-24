import cv2
import matplotlib.pyplot as plt

# Read image
image_path = r'Changing_Shape\puppy.jpg'
image = cv2.imread(image_path)

# Convert image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert image to XYZ
xyz_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2XYZ)

# Convert image to HSV
hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

# Convert image to LAB
lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

# Convert image to YPbPr
ypbpr_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)

# Convert image to YUV
yuv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)


# Display images
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))

axs[0, 0].imshow(image_rgb)
axs[0, 0].set_title('RGB')

axs[0, 1].imshow(xyz_image)
axs[0, 1].set_title('XYZ')

axs[0, 2].imshow(hsv_image)
axs[0, 2].set_title('HSV')

axs[1, 0].imshow(lab_image)
axs[1, 0].set_title('LAB')


axs[1, 1].imshow(ypbpr_image)
axs[1, 1].set_title('YPbPr')

axs[1, 2].imshow(yuv_image)
axs[1, 2].set_title('YUV')


plt.tight_layout()
plt.show()
