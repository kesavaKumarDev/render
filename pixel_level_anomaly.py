# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# def load_and_preprocess(image_path):
#     """Load and preprocess a receipt image with error handling."""
#     # Check if file exists
#     if not os.path.isfile(image_path):
#         raise FileNotFoundError(f"Image file not found: {image_path}")
    
#     # Load image
#     img = cv2.imread(image_path)
    
#     # Check if image was loaded successfully
#     if img is None:
#         raise ValueError(f"Failed to load image: {image_path}")
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Apply adaptive thresholding to handle varying illumination
#     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                   cv2.THRESH_BINARY, 11, 2)
    
#     # Denoise the image
#     denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
#     return img, gray, denoised

# def detect_anomalies_statistical(image, window_size=5, threshold=2.5):
#     """Detect anomalies using statistical approach (Z-score)."""
#     # Create a sliding window
#     height, width = image.shape
#     padding = window_size // 2
#     padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, 
#                                       cv2.BORDER_REPLICATE)
    
#     # Initialize the anomaly mask
#     anomaly_mask = np.zeros_like(image)
    
#     # For large images, process in chunks to avoid memory issues
#     chunk_size = 100
    
#     for chunk_y in range(0, height, chunk_size):
#         end_y = min(chunk_y + chunk_size, height)
#         for chunk_x in range(0, width, chunk_size):
#             end_x = min(chunk_x + chunk_size, width)
            
#             # Process each pixel in the chunk
#             for i in range(chunk_y, end_y):
#                 for j in range(chunk_x, end_x):
#                     # Extract the window
#                     window = padded_image[i:i+window_size, j:j+window_size]
                    
#                     # Calculate local statistics
#                     mean = np.mean(window)
#                     std = np.std(window)
                    
#                     # Get the center pixel
#                     center_pixel = image[i, j]
                    
#                     # Calculate Z-score
#                     if std > 0:  # Avoid division by zero
#                         z_score = abs((center_pixel - mean) / std)
                        
#                         # Mark as anomaly if Z-score is above threshold
#                         if z_score > threshold:
#                             anomaly_mask[i, j] = 255
    
#     return anomaly_mask

# def detect_anomalies_gradient(image, threshold=30):
#     """Detect anomalies based on gradient analysis."""
#     # Calculate gradients
#     grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
#     grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
#     # Calculate gradient magnitude
#     grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
#     # Normalize gradient magnitude
#     grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
#     # Threshold to find areas with high gradient (potential anomalies)
#     _, anomaly_mask = cv2.threshold(grad_mag, threshold, 255, cv2.THRESH_BINARY)
    
#     return anomaly_mask

# def detect_anomalies_histogram(image, block_size=16, threshold=0.3):
#     """Detect anomalies based on histogram analysis."""
#     height, width = image.shape
#     anomaly_mask = np.zeros_like(image)
    
#     # Compute global histogram
#     global_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
#     global_hist = global_hist / np.sum(global_hist) if np.sum(global_hist) > 0 else global_hist
    
#     # Divide the image into blocks
#     for y in range(0, height, block_size):
#         for x in range(0, width, block_size):
#             # Extract the block
#             y_end = min(y + block_size, height)
#             x_end = min(x + block_size, width)
#             block = image[y:y_end, x:x_end]
            
#             # Skip very small blocks
#             if block.size < 25:
#                 continue
                
#             # Calculate histogram of the block
#             local_hist = cv2.calcHist([block], [0], None, [256], [0, 256])
            
#             # Skip empty blocks
#             if np.sum(local_hist) == 0:
#                 continue
                
#             local_hist = local_hist / np.sum(local_hist)
            
#             # Compare histograms using Bhattacharyya distance
#             # Clip histograms to avoid division by zero
#             epsilon = 1e-10
#             global_hist_clipped = np.clip(global_hist, epsilon, None)
#             local_hist_clipped = np.clip(local_hist, epsilon, None)
            
#             # Calculate Bhattacharyya distance
#             hist_diff = cv2.compareHist(global_hist_clipped, local_hist_clipped, cv2.HISTCMP_BHATTACHARYYA)
            
#             # Mark as anomaly if difference is above threshold
#             if hist_diff > threshold:
#                 anomaly_mask[y:y_end, x:x_end] = 255
    
#     return anomaly_mask

# def detect_receipt_anomalies(image_path):
#     """Main function to detect anomalies in a receipt image."""
#     try:
#         # Load and preprocess test image
#         print(f"Loading image from: {image_path}")
#         original, gray, processed = load_and_preprocess(image_path)
        
#         results = {}
        
#         # Use gradient analysis (fastest method)
#         print("Applying gradient analysis...")
#         grad_mask = detect_anomalies_gradient(gray)
#         results['gradient'] = grad_mask
        
#         # Use histogram analysis
#         print("Applying histogram analysis...")
#         hist_mask = detect_anomalies_histogram(gray)
#         results['histogram'] = hist_mask
        
#         # Statistical analysis is computationally intensive, so make it optional
#         run_statistical = False
#         if run_statistical:
#             print("Applying statistical analysis (this may take a while)...")
#             stat_mask = detect_anomalies_statistical(gray)
#             results['statistical'] = stat_mask
            
#             # Include statistical in combined if available
#             combined_mask = cv2.bitwise_or(grad_mask, hist_mask)
#             combined_mask = cv2.bitwise_or(combined_mask, stat_mask)
#         else:
#             # Just combine gradient and histogram
#             combined_mask = cv2.bitwise_or(grad_mask, hist_mask)
        
#         # Clean up with morphological operations
#         kernel = np.ones((3, 3), np.uint8)
#         combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
#         combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
#         results['combined'] = combined_mask
        
#         return original, results
        
#     except Exception as e:
#         print(f"Error in anomaly detection: {str(e)}")
#         raise

# def visualize_results(original, results):
#     """Visualize the detected anomalies."""
#     plt.figure(figsize=(12, 8))
    
#     # Original image
#     plt.subplot(2, 2, 1)
#     plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
#     plt.title('Original Receipt')
#     plt.axis('off')
    
#     # Gradient anomalies
#     plt.subplot(2, 2, 2)
#     plt.imshow(results['gradient'], cmap='gray')
#     plt.title('Gradient Anomalies')
#     plt.axis('off')
    
#     # Histogram anomalies
#     plt.subplot(2, 2, 3)
#     plt.imshow(results['histogram'], cmap='gray')
#     plt.title('Histogram Anomalies')
#     plt.axis('off')
    
#     # Combined anomalies with highlighting
#     highlighted = original.copy()
#     anomaly_mask_color = cv2.cvtColor(results['combined'], cv2.COLOR_GRAY2BGR)
#     anomaly_mask_color[results['combined'] > 0] = [0, 0, 255]  # Red color for anomalies
#     highlighted = cv2.addWeighted(highlighted, 0.7, anomaly_mask_color, 0.3, 0)
    
#     plt.subplot(2, 2, 4)
#     plt.imshow(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))
#     plt.title('Highlighted Anomalies')
#     plt.axis('off')
    
#     plt.tight_layout()
#     plt.show()


# def summarize_anomalies(results):
#     """Summarize detected anomalies as short bullet points."""
#     summary = []

#     if np.any(results['gradient'] > 0):
#         summary.append("⚠️ High gradient variations detected – Possible edge tampering or printing defects.")

#     if np.any(results['histogram'] > 0):
#         summary.append("📊 Histogram inconsistencies found – Potential contrast manipulation or print inconsistencies.")

#     if 'statistical' in results and np.any(results['statistical'] > 0):
#         summary.append("📈 Statistical anomalies detected – Unusual pixel intensity variations.")

#     if not summary:
#         summary.append("✅ No significant anomalies detected.")

#     return summary

# # Example usage in main



# # Example usage with error handling
# if __name__ == "__main__":
#     # Use absolute path to the image file
#     # Replace with your actual receipt image path
#     receipt_path = "../images/pixel_tampered.jpg"  # Update this path
    
#     try:
#         print("Detecting anomalies in receipt image...")
#         original, results = detect_receipt_anomalies(receipt_path)
        
#         # Summarize findings
#         anomaly_summary = summarize_anomalies(results)
        
#         # Print observations
#         print("\n📝 **Anomaly Observations:**")
#         for point in anomaly_summary:
#             print(f"- {point}")
        
#         print("\nVisualizing results...")
#         visualize_results(original, results)
        
#     except FileNotFoundError as e:
#         print(f"File not found error: {str(e)}")
#         print("Please check the file path and ensure the image exists.")
        
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")


from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class ReceiptTamperingDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original, self.gray, self.processed = self.load_and_preprocess()
        self.results = {}
    
    def load_and_preprocess(self):
        if not os.path.isfile(self.image_path):
            raise FileNotFoundError(f"Image file not found: {self.image_path}")
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {self.image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        return img, gray, denoised
    
    def detect_anomalies_gradient(self, threshold=30):
        grad_x = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, anomaly_mask = cv2.threshold(grad_mag, threshold, 255, cv2.THRESH_BINARY)
        self.results['gradient'] = anomaly_mask
    
    def detect_anomalies_histogram(self, block_size=16, threshold=0.3):
        height, width = self.gray.shape
        anomaly_mask = np.zeros_like(self.gray)
        global_hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
        global_hist = global_hist / np.sum(global_hist) if np.sum(global_hist) > 0 else global_hist
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                y_end = min(y + block_size, height)
                x_end = min(x + block_size, width)
                block = self.gray[y:y_end, x:x_end]
                if block.size < 25:
                    continue
                local_hist = cv2.calcHist([block], [0], None, [256], [0, 256])
                if np.sum(local_hist) == 0:
                    continue
                local_hist = local_hist / np.sum(local_hist)
                hist_diff = cv2.compareHist(global_hist, local_hist, cv2.HISTCMP_BHATTACHARYYA)
                if hist_diff > threshold:
                    anomaly_mask[y:y_end, x:x_end] = 255
        self.results['histogram'] = anomaly_mask
    
    def detect_anomalies(self):
        self.detect_anomalies_gradient()
        self.detect_anomalies_histogram()
        combined_mask = cv2.bitwise_or(self.results['gradient'], self.results['histogram'])
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        self.results['combined'] = combined_mask
        return self.results
    
    def summarize_anomalies(self):
        summary = []
        if np.any(self.results['gradient'] > 0):
            summary.append("⚠️ High gradient variations detected – Possible edge tampering or printing defects.")
        if np.any(self.results['histogram'] > 0):
            summary.append("📊 Histogram inconsistencies found – Potential contrast manipulation or print inconsistencies.")
        if not summary:
            summary.append("✅ No significant anomalies detected.")
        return summary
    
"""
Helps detect copy-paste artifacts or blending issues.

Useful to detect lighting inconsistencies, color mismatches, or fake overlays.

Gradient variations and histogram inconsistencies were used to identify potential tampered regions in receipts, such as altered totals or inserted items
"""