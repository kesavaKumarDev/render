# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# def detect_forgery_sift(image_path):
#     """Uses SIFT for copy-move forgery detection."""
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     sift = cv2.SIFT_create()

#     keypoints, descriptors = sift.detectAndCompute(image, None)
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(descriptors, descriptors, k=2)

#     # Apply Lowe's Ratio Test to filter matches
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good_matches.append(m)

#     result_img = cv2.drawMatches(image, keypoints, image, keypoints, good_matches[:30], None, flags=2)

#     plt.figure(figsize=(12, 6))
#     plt.imshow(result_img, cmap='gray')
#     plt.title("Copy-Move Forgery Detection (SIFT Feature Matching)")
#     plt.show()

# # Example usage
# detect_forgery_sift("../images/processed_receipt.jpg")

"""
   If keypoints also cluster in text regions (prices, dates, totals) → Possible manipulation.
   If keypoints mostly appear in non-text regions (edges, background) → Likely not tampered.
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt
from flask import Flask, request, jsonify, send_file
import os
import io
import uuid

class CloneDetector:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

    def detect_forgery_sift(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        
        if descriptors is None:
            return None, "No keypoints detected."
        
        matches = self.bf.knnMatch(descriptors, descriptors, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        result_img = cv2.drawMatches(image, keypoints, image, keypoints, good_matches[:30], None, flags=2)
        return result_img, f"Detected {len(good_matches)} potential forged regions."