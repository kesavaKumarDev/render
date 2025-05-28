import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.filters import threshold_local
from sklearn.ensemble import IsolationForest

def detect_tampering_ml(image_path):
    # Check if file exists
    if not os.path.isfile(image_path):
        print(f"Error: File not found at {image_path}")
        return None, [], None
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if image was loaded successfully
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        print("File may be corrupted or in an unsupported format")
        return None, [], None
    
    # Convert to RGB for display with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Feature extraction methods
    features = []
    
    # 1. Error Level Analysis (ELA)
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    saved_image = cv2.imread(temp_path)
    saved_gray = cv2.cvtColor(saved_image, cv2.COLOR_BGR2GRAY)
    ela = cv2.absdiff(gray, saved_gray)
    features.append(ela)
    
    # 2. Noise analysis
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = cv2.absdiff(gray, blur)
    features.append(noise)
    
    # 3. Edge detection
    edges = cv2.Canny(gray, 100, 200)
    features.append(edges)
    
    # 4. Texture analysis (Local Binary Patterns)
    def get_pixel(img, center, x, y):
        new_value = 0
        try:
            if img[x][y] >= center:
                new_value = 1
        except:
            pass
        return new_value

    def lbp_calculated_pixel(img, x, y):
        center = img[x][y]
        val_ar = []
        val_ar.append(get_pixel(img, center, x-1, y-1))
        val_ar.append(get_pixel(img, center, x-1, y))
        val_ar.append(get_pixel(img, center, x-1, y+1))
        val_ar.append(get_pixel(img, center, x, y+1))
        val_ar.append(get_pixel(img, center, x+1, y+1))
        val_ar.append(get_pixel(img, center, x+1, y))
        val_ar.append(get_pixel(img, center, x+1, y-1))
        val_ar.append(get_pixel(img, center, x, y-1))
        power_val = [1, 2, 4, 8, 16, 32, 64, 128]
        val = 0
        for i in range(len(val_ar)):
            val += val_ar[i] * power_val[i]
        return val

    height, width = gray.shape
    lbp_image = np.zeros((height, width), np.uint8)
    for i in range(1, height-1):
        for j in range(1, width-1):
            lbp_image[i, j] = lbp_calculated_pixel(gray, i, j)
    features.append(lbp_image)
    
    # Combine features into an anomaly detection input
    feature_vectors = []
    for i in range(height):
        for j in range(width):
            feature_vector = [f[i, j] for f in features]
            feature_vectors.append(feature_vector)
    
    # Apply anomaly detection
    X = np.array(feature_vectors)
    clf = IsolationForest(contamination=0.05, random_state=42)
    predictions = clf.fit_predict(X)
    
    # Reshape predictions to image dimensions
    anomaly_map = predictions.reshape(height, width)
    anomaly_map = np.where(anomaly_map == -1, 255, 0).astype(np.uint8)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    anomaly_map = cv2.morphologyEx(anomaly_map, cv2.MORPH_OPEN, kernel)
    anomaly_map = cv2.morphologyEx(anomaly_map, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the anomaly map
    contours, _ = cv2.findContours(anomaly_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    min_area = 100
    suspicious_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            suspicious_regions.append(contour)
    
    # Draw rectangles around suspicious regions
    result_image = image_rgb.copy()
    for contour in suspicious_regions:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Clean up the temporary file
    try:
        os.remove(temp_path)
    except:
        pass
    
    return result_image, suspicious_regions, anomaly_map

def analyze_tampering(image_path):
    """Comprehensive analysis of a potentially tampered receipt"""
    print(f"Analyzing image: {image_path}")
    
    result = detect_tampering_ml(image_path)
    if result[0] is None:
        return None
    
    result_image, regions, anomaly_map = result
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 3, 1)
    original = cv2.imread(image_path)
    if original is not None:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Original Receipt")
    else:
        plt.title("Original Receipt (Failed to Load)")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(anomaly_map, cmap='hot')
    plt.title("Anomaly Heat Map")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(result_image)
    plt.title(f"Detected {len(regions)} Suspicious Regions")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("tamper_analysis_result.png")
    plt.show()
    
    # Detailed report
    print(f"\nTamper Analysis Report for {image_path}")
    print(f"Number of suspicious regions detected: {len(regions)}")
    
    if len(regions) > 0:
        print("\nSuspicious Region Details:")
        for i, contour in enumerate(regions):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            print(f"Region {i+1}: Position (x={x}, y={y}), Size {w}x{h}, Area {area} pixels")
    else:
        print("No tampering detected with current thresholds.")
    
    return result_image

# Simple version without machine learning, using only ELA
def simple_tampering_detection(image_path):
    """A simpler version that uses only Error Level Analysis"""
    # Check if file exists
    if not os.path.isfile(image_path):
        print(f"Error: File not found at {image_path}")
        return None
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if image was loaded successfully
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        print("File may be corrupted or in an unsupported format")
        return None
    
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform Error Level Analysis (ELA)
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    saved_image = cv2.imread(temp_path)
    saved_gray = cv2.cvtColor(saved_image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the difference
    ela = cv2.absdiff(gray, saved_gray)
    
    # Enhance the difference
    ela_enhanced = cv2.equalizeHist(ela)
    
    # Threshold to find suspicious areas
    _, thresh = cv2.threshold(ela_enhanced, 30, 255, cv2.THRESH_BINARY)
    
    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by size
    min_area = 100
    suspicious_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            suspicious_regions.append(contour)
    
    # Draw boxes
    result = image_rgb.copy()
    for contour in suspicious_regions:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Clean up
    try:
        os.remove(temp_path)
    except:
        pass
    
    # Display results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Receipt")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(ela, cmap='gray')
    plt.title("Error Level Analysis")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(thresh, cmap='gray')
    plt.title("Thresholded Anomalies")
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(result)
    plt.title(f"Detected {len(suspicious_regions)} Suspicious Regions")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("simple_tamper_analysis.png")
    plt.show()
    
    return result

# Usage example
if __name__ == "__main__":
    # You can try both methods
    try:
        image_path = input("Enter the path to the receipt image: ")
        
        print("\nRunning simple detection...")
        simple_result = simple_tampering_detection(image_path)
        
        # print("\nRunning advanced ML-based detection...")
        # advanced_result = analyze_tampering(image_path)
        
        if simple_result is None :
            print("\nFailed to analyze the image. Please check the file path and format.")
        else:
            print("\nAnalysis complete. Results saved as PNG files.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()