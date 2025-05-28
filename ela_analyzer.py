# import cv2
# import numpy as np
# from PIL import Image, ImageChops, ImageEnhance
# import matplotlib.pyplot as plt

# def error_level_analysis(image_path, quality=90):
#     """
#     Perform Error Level Analysis (ELA) to detect image manipulation.
    
#     Args:
#     - image_path: Path to the input image.
#     - quality: JPEG re-saving quality (default: 90).
    
#     Returns:
#     - ELA image highlighting altered regions.
#     """

#     # Open the original image
#     original = Image.open(image_path).convert("RGB")
    
#     # Save a temporary JPEG version at the given quality
#     temp_path = "temp_compressed.jpg"
#     original.save(temp_path, "JPEG", quality=quality)

#     # Open the re-saved image
#     compressed = Image.open(temp_path)

#     # Compute the absolute difference (ELA map)
#     ela_image = ImageChops.difference(original, compressed)

#     # Enhance the differences to make them more visible
#     extrema = ela_image.getextrema()
#     max_diff = max([ex[1] for ex in extrema])
#     if max_diff == 0:
#         max_diff = 1  # Avoid division by zero

#     scale = 255.0 / max_diff
#     ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

#     return ela_image

# def detect_suspicious_regions(ela_image):
#     ela_np = np.array(ela_image.convert("L"))  # Convert to grayscale
#     _, thresh = cv2.threshold(ela_np, 150, 255, cv2.THRESH_BINARY)

#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     return contours

# def show_ela(image_path):
#     """
#     Display the original and ELA images side by side.
#     """
#     original = Image.open(image_path).convert("RGB")
#     ela_img = error_level_analysis(image_path)

#     contours = detect_suspicious_regions(ela_img)
#     print(f"Suspicious regions detected: {len(contours)}")

#     # Convert images to NumPy for visualization
#     original_np = np.array(original)
#     ela_np = np.array(ela_img)

#     # Display images
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#     axes[0].imshow(original_np)
#     axes[0].set_title("Original Image")
#     axes[0].axis("off")

#     axes[1].imshow(ela_np, cmap="gray")
#     axes[1].set_title("Error Level Analysis (ELA)")
#     axes[1].axis("off")

#     plt.show()


# # Example Usage:
# image_path = "../images/processed_receipt.jpg"  # Change this to your receipt image
# show_ela(image_path)

"""
An untouched image has consistent error levels.Text and background have similar intensity in ELA.

Altered regions have higher error levels.Edges and text show up as bright spots in ELA.

Natural objects (e.g., receipts, printed text) should have smooth ELA transitions.Abrupt changes indicate pasting or tampering.
"""




# ela_analyzer.py

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import os
import tempfile

class ErrorLevelAnalyzer:
    """
    A class for performing Error Level Analysis (ELA) to detect image manipulation.
    """
    
    def __init__(self, quality=90):
        """
        Initialize the ELA analyzer.
        
        Args:
            quality (int): JPEG re-saving quality (default: 90).
        """
        self.quality = quality
    
    def analyze(self, image_path):
        """
        Perform Error Level Analysis on an image.
        
        Args:
            image_path (str): Path to the input image.
            
        Returns:
            PIL.Image: ELA image highlighting altered regions.
        """
        # Open the original image
        original = Image.open(image_path).convert("RGB")
        
        # Create a temporary file for the compressed version
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Save a temporary JPEG version at the given quality
        original.save(temp_path, "JPEG", quality=self.quality)
        
        # Open the re-saved image
        compressed = Image.open(temp_path)
        
        # Compute the absolute difference (ELA map)
        ela_image = ImageChops.difference(original, compressed)
        
        # Enhance the differences to make them more visible
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1  # Avoid division by zero
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return ela_image
    
    def detect_suspicious_regions(self, ela_image, threshold=150):
        """
        Detect suspicious regions in the ELA image.
        
        Args:
            ela_image (PIL.Image): ELA image from analyze method.
            threshold (int): Threshold for binary conversion (default: 150).
            
        Returns:
            list: List of contours representing suspicious regions.
        """
        ela_np = np.array(ela_image.convert("L"))  # Convert to grayscale
        _, thresh = cv2.threshold(ela_np, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def get_analysis_results(self, image_path, threshold=150):
        """
        Get complete analysis results.
        
        Args:
            image_path (str): Path to the input image.
            threshold (int): Threshold for binary conversion (default: 150).
            
        Returns:
            dict: Dictionary containing analysis results.
        """
        original = Image.open(image_path).convert("RGB")
        ela_image = self.analyze(image_path)
        contours = self.detect_suspicious_regions(ela_image, threshold)
        
        return {
            "original_image": original,
            "ela_image": ela_image,
            "suspicious_regions_count": len(contours),
            "suspicious_regions": contours
        }


