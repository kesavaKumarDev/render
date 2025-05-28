# import cv2
# import numpy as np
# import pywt
# import matplotlib.pyplot as plt
# from scipy.fftpack import dct, idct
# from skimage.restoration import estimate_sigma
# from skimage.filters import laplace

# # Noise Pattern Analysis using PCA-based Noise Estimation
# def analyze_noise_pattern(image_path, threshold=5.0):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     noise_sigma = estimate_sigma(image, channel_axis=None, average_sigmas=True)
    
#     plt.imshow(image, cmap='gray')
#     plt.title(f"Noise Pattern Analysis (σ = {noise_sigma:.2f})")
#     plt.colorbar()
#     plt.show()
    
#     anomaly = noise_sigma > threshold
#     return noise_sigma, anomaly

# # Shadow & Lighting Consistency Check using Retinex-based Analysis
# def check_lighting_consistency(image_path, threshold=50):
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     log_image = np.log1p(gray.astype(np.float32))
#     filtered = cv2.GaussianBlur(log_image, (15, 15), 0)
#     illumination_map = np.exp(filtered)
#     corrected_image = (gray / (illumination_map + 1e-6)) * 255
#     corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
    
#     laplacian_var = cv2.Laplacian(corrected_image, cv2.CV_64F).var()
    
#     plt.imshow(corrected_image, cmap='gray')
#     plt.title("Lighting Consistency Check (Retinex)")
#     plt.colorbar()
#     plt.show()
    
#     anomaly = laplacian_var < threshold
#     return corrected_image, anomaly

# # JPEG Ghost Detection using DCT Block Discrepancy Analysis
# def jpeg_ghost_detection(image_path, quality=95, threshold=20):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     dct_coeffs = dct(dct(image.T, norm='ortho').T, norm='ortho')
#     compressed_image = np.round(idct(idct(dct_coeffs.T, norm='ortho').T, norm='ortho')).astype(np.uint8)
#     difference = cv2.absdiff(image, compressed_image)
#     avg_diff = np.mean(difference)
    
#     plt.imshow(difference, cmap='hot')
#     plt.title("JPEG Ghost Detection (DCT Analysis)")
#     plt.colorbar()
#     plt.show()
    
#     anomaly = avg_diff > threshold
#     return difference, anomaly

# # Automated Tampering Detection
# def detect_tampering(image_path):
#     noise_sigma, noise_anomaly = analyze_noise_pattern(image_path)
#     _, lighting_anomaly = check_lighting_consistency(image_path)
#     _, jpeg_anomaly = jpeg_ghost_detection(image_path)
    
#     if noise_anomaly or lighting_anomaly or jpeg_anomaly:
#         print("⚠️ Possible tampering detected!")
#     else:
#         print("✅ No significant tampering detected.")
    
#     return {
#         "noise_sigma": noise_sigma,
#         "noise_anomaly": noise_anomaly,
#         "lighting_anomaly": lighting_anomaly,
#         "jpeg_anomaly": jpeg_anomaly
#     }

# # Example Usage
# image_path = '../images/processed_receipt.jpg'  # Replace with actual image path
# detection_results = detect_tampering(image_path)


# 1️⃣ Noise Pattern Analysis:
#    - Checks for uniform noise distribution.
#    - Sudden noise variations suggest possible modifications.

# 2️⃣ Shadow & Lighting Consistency:
#    - Examines illumination uniformity across the receipt.
#    - Unnatural lighting shifts may indicate splicing or manipulation.

# 3️⃣ JPEG Ghost Detection:
#    - Analyzes compression inconsistencies.
#    - Strong intensity differences in the heatmap suggest potential tampering.


import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage.restoration import estimate_sigma
import warnings

class ForgeryDetector:
    def __init__(self, threshold_noise=5.0, threshold_lighting=50, threshold_jpeg=20):
        self.threshold_noise = threshold_noise
        self.threshold_lighting = threshold_lighting
        self.threshold_jpeg = threshold_jpeg

    def analyze_noise_pattern(self, image):
        # Ensure grayscale conversion
        if len(image.shape) == 3:  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Estimate noise
        noise_sigma = estimate_sigma(image, channel_axis=None, average_sigmas=True)
        anomaly = noise_sigma > self.threshold_noise
        return noise_sigma, anomaly

    def check_lighting_consistency(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        log_image = np.log1p(gray.astype(np.float32))  # Prevent log(0) errors
        filtered = cv2.GaussianBlur(log_image, (15, 15), 0)
        illumination_map = np.exp(filtered)
        corrected_image = np.clip((gray / (illumination_map + 1e-6)) * 255, 0, 255).astype(np.uint8)
        laplacian_var = cv2.Laplacian(corrected_image, cv2.CV_64F).var()
        anomaly = laplacian_var < self.threshold_lighting
        return anomaly

    def jpeg_ghost_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply DCT and IDCT with error handling
        try:
            dct_coeffs = dct(dct(gray.T, norm='ortho').T, norm='ortho')
            compressed_image = np.round(idct(idct(dct_coeffs.T, norm='ortho').T, norm='ortho')).astype(np.uint8)
        except Exception as e:
            warnings.warn(f"DCT/IDCT processing failed: {e}")
            return False  # Assume no anomaly if processing fails
        
        difference = cv2.absdiff(gray, compressed_image)
        avg_diff = np.mean(difference)
        anomaly = avg_diff > self.threshold_jpeg
        return anomaly

    def detect_tampering(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Failed to read image"}

        noise_sigma, noise_anomaly = self.analyze_noise_pattern(image)
        lighting_anomaly = self.check_lighting_consistency(image)
        jpeg_anomaly = self.jpeg_ghost_detection(image)

        result = {
            "noise_sigma": noise_sigma,
            "noise_anomaly": noise_anomaly,
            "lighting_anomaly": lighting_anomaly,
            "jpeg_anomaly": jpeg_anomaly,
            "tampering_detected": noise_anomaly or lighting_anomaly or jpeg_anomaly
        }
        return result

