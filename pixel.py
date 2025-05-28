import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import img_as_float, img_as_ubyte
from skimage.feature import local_binary_pattern
from skimage.exposure import equalize_hist
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

def pixel_level_anomaly_detection(image_path, sensitivity=0.5):
    """
    Detect pixel-level anomalies in receipt images with adjustable sensitivity.
    
    Parameters:
    -----------
    image_path : str
        Path to the input receipt image
    sensitivity : float
        Value between 0 and 1 controlling detection sensitivity (higher = more sensitive)
    
    Returns:
    --------
    tuple: (original image, anomaly map, highlighted image, suspicious regions)
    """
    # Validate input file
    if not os.path.isfile(image_path):
        print(f"Error: File not found at {image_path}")
        return None, None, None, []
    
    # Read the image
    original = cv2.imread(image_path)
    if original is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None, None, []
    
    # Convert to RGB for visualization
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Image preprocessing
    # Normalize lighting and contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Multi-level feature extraction
    features = []
    
    # 1. Error Level Analysis (ELA)
    # Save and reload the image at different quality levels
    quality_levels = [90, 80, 70]
    ela_features = []
    
    for quality in quality_levels:
        temp_path = f"temp_q{quality}.jpg"
        cv2.imwrite(temp_path, original, [cv2.IMWRITE_JPEG_QUALITY, quality])
        reloaded = cv2.imread(temp_path)
        reloaded_gray = cv2.cvtColor(reloaded, cv2.COLOR_BGR2GRAY)
        ela_diff = cv2.absdiff(gray, reloaded_gray)
        ela_features.append(ela_diff)
        
        try:
            os.remove(temp_path)
        except:
            pass
    
    # Combine ELA features with different weights
    ela = ela_features[0] * 0.5 + ela_features[1] * 0.3 + ela_features[2] * 0.2
    ela = equalize_hist(ela).astype(np.float32)
    features.append(ela)
    
    # 2. Multi-scale noise analysis
    noise_features = []
    sigma_values = [1, 2, 3]
    
    for sigma in sigma_values:
        # Apply Gaussian blur
        blurred = gaussian_filter(gray, sigma=sigma)
        # Extract noise
        noise = cv2.absdiff(gray, blurred).astype(np.float32) / 255.0
        noise_features.append(noise)
    
    # Combine noise features
    noise = np.maximum.reduce(noise_features)
    features.append(noise)
    
    # 3. Texture anomalies using Local Binary Patterns
    # Multi-radius LBP for capturing both fine and coarse textures
    lbp_features = []
    radii = [1, 2, 3]
    n_points = 8 * np.array(radii, dtype=np.int32)
    
    for i, radius in enumerate(radii):
        lbp = local_binary_pattern(gray, n_points[i], radius, method='uniform')
        lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-10)  # Normalize
        lbp_features.append(lbp)
    
    # Combine LBP features
    lbp_combined = np.mean(lbp_features, axis=0)
    features.append(lbp_combined)
    
    # 4. Edge inconsistency detection
    # Multiple edge detection methods
    edges_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    edges_sobel = np.abs(edges_sobel)
    edges_sobel = (edges_sobel - edges_sobel.min()) / (edges_sobel.max() - edges_sobel.min() + 1e-10)
    
    edges_canny = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    
    edges_laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    edges_laplacian = (edges_laplacian - edges_laplacian.min()) / (edges_laplacian.max() - edges_laplacian.min() + 1e-10)
    
    # Combine edge features
    edges_combined = edges_sobel * 0.3 + edges_canny * 0.4 + edges_laplacian * 0.3
    features.append(edges_combined)
    
    # 5. Frequency domain analysis (DCT)
    h, w = gray.shape
    # Expand to optimal size for DCT
    dct_h, dct_w = 2**int(np.ceil(np.log2(h))), 2**int(np.ceil(np.log2(w)))
    dct_img = np.zeros((dct_h, dct_w))
    dct_img[:h, :w] = gray
    
    # Apply DCT
    dct = cv2.dct(dct_img.astype(np.float32))
    # Log scale for better visualization
    dct_log = np.log(np.abs(dct) + 1)
    # Normalize
    dct_norm = (dct_log - dct_log.min()) / (dct_log.max() - dct_log.min() + 1e-10)
    # Crop back to original size
    dct_norm = dct_norm[:h, :w]
    features.append(dct_norm)
    
    # Build pixel-level feature vectors
    height, width = gray.shape
    feature_array = np.zeros((height, width, len(features)))
    
    for i, feature in enumerate(features):
        # Ensure all features have the same shape
        feature_resized = cv2.resize(feature, (width, height))
        feature_array[:, :, i] = feature_resized
    
    # Reshape for anomaly detection
    n_samples = height * width
    X = feature_array.reshape((n_samples, len(features)))
    
    # Adjust the isolation forest contamination based on sensitivity
    contamination = 0.02 + sensitivity * 0.08  # Range from 0.02 to 0.1
    
    # Apply Isolation Forest for anomaly detection
    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    y_pred = clf.fit_predict(X)
    
    # DBSCAN for spatial clustering of anomalies
    # Create spatial features (x, y coordinates + anomaly score)
    anomaly_scores = clf.score_samples(X)
    anomaly_scores = -anomaly_scores  # Convert to anomaly score (higher = more anomalous)
    
    # Normalize anomaly scores to [0,1]
    anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-10)
    
    # Create feature array with coordinates and anomaly score
    coords = np.zeros((n_samples, 3))
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            coords[idx, 0] = j / width  # x-coordinate (normalized)
            coords[idx, 1] = i / height  # y-coordinate (normalized)
            coords[idx, 2] = anomaly_scores[idx] * 5  # Weighted anomaly score
    
    # Apply DBSCAN to find clusters of anomalies
    eps_value = 0.02 - sensitivity * 0.015  # Adjust eps based on sensitivity
    min_samples = int(5 + (1 - sensitivity) * 15)  # Adjust min_samples based on sensitivity
    
    db = DBSCAN(eps=eps_value, min_samples=min_samples).fit(coords)
    labels = db.labels_

    # Create anomaly map
    anomaly_map = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            if labels[idx] != -1:  # If part of a cluster
                anomaly_map[i, j] = anomaly_scores[idx]
            else:
                anomaly_map[i, j] = 0
    
    # Post-processing: apply threshold to clean up weak detections
    threshold = 0.3 - sensitivity * 0.2  # Adjust threshold based on sensitivity
    anomaly_map[anomaly_map < threshold] = 0
    
    # Scale for visualization
    anomaly_map_viz = anomaly_map * 255
    anomaly_map_viz = anomaly_map_viz.astype(np.uint8)
    
    # Apply morphological operations to clean up the anomaly map
    kernel = np.ones((3, 3), np.uint8)
    anomaly_map_morph = cv2.morphologyEx(anomaly_map_viz, cv2.MORPH_OPEN, kernel)
    anomaly_map_morph = cv2.morphologyEx(anomaly_map_morph, cv2.MORPH_CLOSE, kernel)
    
    # Find contours for suspicious regions
    threshold_value = int(50 + sensitivity * 100)  # Dynamic threshold based on sensitivity
    _, thresh = cv2.threshold(anomaly_map_morph, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    min_area = 20 + (1 - sensitivity) * 80  # Dynamic min area based on sensitivity
    suspicious_regions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            suspicious_regions.append(contour)
    
    # Draw bounding boxes on a copy of the original image
    result_image = original_rgb.copy()
    
    for contour in suspicious_regions:
        x, y, w, h = cv2.boundingRect(contour)
        # Add padding to the bounding box
        padding = 5
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        w_pad = min(width - x_pad, w + 2 * padding)
        h_pad = min(height - y_pad, h + 2 * padding)
        
        cv2.rectangle(result_image, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), (255, 0, 0), 2)
        
        # Add confidence text
        confidence = np.mean(anomaly_map[y:y+h, x:x+w]) * 100
        cv2.putText(result_image, f"{confidence:.1f}%", (x_pad, y_pad - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Also create a heatmap overlay
    heatmap = cv2.applyColorMap(anomaly_map_viz, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend original and heatmap
    alpha = 0.7
    highlighted = cv2.addWeighted(original_rgb, alpha, heatmap, 1-alpha, 0)
    
    return original_rgb, anomaly_map_viz, highlighted, suspicious_regions

def visualize_results(original, anomaly_map, highlighted, suspicious_regions):
    """
    Visualize the anomaly detection results
    """
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    plt.imshow(original)
    plt.title("Original Receipt")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(anomaly_map, cmap='hot')
    plt.title("Anomaly Heat Map")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(highlighted)
    plt.title("Anomaly Overlay")
    plt.axis('off')
    
    # Create a copy with bounding boxes
    result = original.copy()
    for contour in suspicious_regions:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    plt.subplot(2, 2, 4)
    plt.imshow(result)
    plt.title(f"Detected {len(suspicious_regions)} Suspicious Regions")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("receipt_anomaly_detection.png", dpi=300)
    plt.show()
    
    return result

def analyze_tampering_with_sensitivity(image_path, sensitivity=0.5):
    """
    Analyze receipt tampering with adjustable sensitivity
    
    Parameters:
    -----------
    image_path : str
        Path to the receipt image
    sensitivity : float
        Value between 0 and 1 controlling detection sensitivity (higher = more sensitive)
    """
    print(f"Analyzing image: {image_path} with sensitivity {sensitivity}")
    
    # Run pixel-level anomaly detection
    original, anomaly_map, highlighted, suspicious_regions = pixel_level_anomaly_detection(
        image_path, sensitivity)
    
    if original is None:
        print("Analysis failed. Please check the image file.")
        return None
    
    # Visualize results
    result = visualize_results(original, anomaly_map, highlighted, suspicious_regions)
    
    # Print detailed report
    print(f"\nTamper Analysis Report for {image_path}")
    print(f"Sensitivity level: {sensitivity * 100:.1f}%")
    print(f"Number of suspicious regions detected: {len(suspicious_regions)}")
    
    if len(suspicious_regions) > 0:
        print("\nSuspicious Region Details:")
        for i, contour in enumerate(suspicious_regions):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Calculate confidence score based on anomaly values in the region
            confidence = np.mean(anomaly_map[y:y+h, x:x+w]) / 255 * 100
            
            print(f"Region {i+1}: Position (x={x}, y={y}), Size {w}x{h}, Area {area} pixels, Confidence: {confidence:.1f}%")
            
            # Analyze potential type of tampering
            if w < 20 and h < 20:
                print(f"  - Possible type: Small pixel manipulation")
            elif w > 100 and h < 30:
                print(f"  - Possible type: Text/number alteration")
            elif w > 50 and h > 50:
                print(f"  - Possible type: Image splicing or content replacement")
            else:
                print(f"  - Possible type: Unknown manipulation")
    else:
        print("No tampering detected with current sensitivity settings.")
    
    return result

# If run as a script
if __name__ == "__main__":
    try:
        image_path = input("Enter the path to the receipt image: ")
        sensitivity = float(input("Enter sensitivity (0.0-1.0, where 1.0 is most sensitive): "))
        
        if sensitivity < 0 or sensitivity > 1:
            print("Sensitivity must be between 0.0 and 1.0. Setting to default 0.5.")
            sensitivity = 0.5
            
        analyze_tampering_with_sensitivity(image_path, sensitivity)
        
    except ValueError:
        print("Invalid sensitivity value. Using default of 0.5.")
        analyze_tampering_with_sensitivity(image_path, 0.5)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()