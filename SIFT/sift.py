import cv2
import numpy as np
import matplotlib.pyplot as plt
from Sift_Operations import *
Image1 = cv2.resize(cv2.imread(r"SIFT\taj1.jpeg"), (980, 980))
Image2 = cv2.resize(cv2.imread(r"SIFT\taj2.jpeg"), (980, 980))

Image1_gray = cv2.cvtColor(Image1, cv2.COLOR_BGR2GRAY)
Image2_gray = cv2.cvtColor(Image2, cv2.COLOR_BGR2GRAY)
Image1_key_points, Image1_descriptors = extract_sift_features(Image1_gray)
Image2_key_points, Image2_descriptors = extract_sift_features(Image2_gray)

showing_sift_features(Image1_gray, Image1, Image1_key_points)

norm = cv2.NORM_L2
bruteForce = cv2.BFMatcher(norm)
matches = bruteForce.match(Image1_descriptors, Image2_descriptors)

matches = sorted(matches, key=lambda match: match.distance)
matched_img = cv2.drawMatches(
    Image1, Image1_key_points,
    Image2, Image2_key_points,
    matches[:100], Image2.copy())
cv2.imshow("matched keypoints", matched_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
