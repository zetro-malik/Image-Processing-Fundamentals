# To check quality or Structural similarity between 2

from skimage import io
from skimage.metrics import structural_similarity as compare_ssim
# Load images
img1 = io.imread(r'SSIM\test_img.jpg')
img2 = io.imread(r'SSIM\test_img_compressed.jpeg')

# Calculate SSIM
ssim = compare_ssim(img1, img2, multichannel=True)

print("The SSIM value is: ", ssim)
