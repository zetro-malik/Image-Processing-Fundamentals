import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_sift_features(img):
    sift_initialize = cv2.SIFT_create()
    key_points, descriptors = sift_initialize.detectAndCompute(img, None)
    return key_points, descriptors


def showing_sift_features(img1, img2, key_points):
    return cv2.imshow("keypoints", cv2.drawKeypoints(img1, key_points, img2.copy()))
