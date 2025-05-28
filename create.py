import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.util import random_noise
import argparse

def add_salt_pepper_noise(image, amount=0.05):
    """Add salt and pepper noise to image"""
    return random_noise(image, mode='s&p', amount=amount)

def add_gaussian_noise(image, var=0.01):
    """Add gaussian noise to image"""
    return random_noise(image, mode='gaussian', var=var)

def add_speckle_noise(image, var=0.1):
    """Add speckle noise to image"""
    return random_noise(image, mode='speckle', var=var)

def add_poisson_noise(image):
    """Add poisson noise to image"""
    return random_noise(image, mode='poisson')

def add_local_corruption(image, intensity=0.5, num_regions=5, max_size=30):
    """Add local corruptions to random regions of the image"""
    corrupted = image.copy()
    height, width = image.shape[:2]
    
    for _ in range(num_regions):
        # Random position and size
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        size_x = np.random.randint(5, max_size)
        size_y = np.random.randint(5, max_size)
        
        # Make sure region is within image bounds
        x_end = min(x + size_x, width)
        y_end = min(y + size_y, height)
        
        # Apply corruption (random values)
        if len(image.shape) == 3:  # Color image
            corrupted[y:y_end, x:x_end] = np.random.random((y_end-y, x_end-x, image.shape[2])) * intensity + \
                                           corrupted[y:y_end, x:x_end] * (1-intensity)
        else:  # Grayscale image
            corrupted[y:y_end, x:x_end] = np.random.random((y_end-y, x_end-x)) * intensity + \
                                           corrupted[y:y_end, x:x_end] * (1-intensity)
    
    return corrupted

def add_pixel_drift(image, intensity=10):
    """Shift pixel values by a random amount"""
    drift = np.random.randint(-intensity, intensity+1, size=image.shape)
    
    if len(image.shape) == 3:  # Color image
        # Convert to float for calculation
        float_img = image.astype(float)
        # Apply drift
        drifted = float_img + drift * (float_img/255.0)  # Scale drift by pixel intensity
        # Clip to valid range
        return np.clip(drifted, 0, 1)
    else:  # Grayscale
        float_img = image.astype(float)
        drifted = float_img + drift * (float_img/255.0)
        return np.clip(drifted, 0, 1)

def create_dead_pixels(image, percentage=0.01):
    """Create random dead pixels in the image"""
    corrupted = image.copy()
    height, width = image.shape[:2]
    
    # Calculate number of pixels to modify
    num_pixels = int(height * width * percentage)
    
    # Generate random pixel coordinates
    y_coords = np.random.randint(0, height, num_pixels)
    x_coords = np.random.randint(0, width, num_pixels)
    
    # Set pixels to black or white
    for y, x in zip(y_coords, x_coords):
        if np.random.random() > 0.5:  # 50% chance for black or white
            corrupted[y, x] = 0  # Black
        else:
            if len(image.shape) == 3:  # Color image
                corrupted[y, x] = [1, 1, 1]  # White for color image
            else:  # Grayscale
                corrupted[y, x] = 1  # White for grayscale
    
    return corrupted

def add_bit_error(image, bit_planes=1):
    """Simulate bit errors in specific bit planes"""
    corrupted = image.copy()
    
    # Convert to integer representation if needed
    if corrupted.dtype != np.uint8 and np.max(corrupted) <= 1.0:
        corrupted = (corrupted * 255).astype(np.uint8)
    
    # Randomly flip bits in the specified number of bit planes
    for _ in range(bit_planes):
        bit_pos = np.random.randint(0, 8)  # 8 bits in a byte
        bit_mask = 1 << bit_pos
        
        # Random positions to flip (about 5% of pixels)
        mask = np.random.random(corrupted.shape[:2]) < 0.05
        
        # Apply XOR operation to flip bits
        if len(corrupted.shape) == 3:  # Color image
            for c in range(corrupted.shape[2]):
                corrupted[mask, c] = corrupted[mask, c] ^ bit_mask
        else:  # Grayscale
            corrupted[mask] = corrupted[mask] ^ bit_mask
    
    # Convert back to float if needed
    if image.dtype != np.uint8 and np.max(image) <= 1.0:
        corrupted = corrupted.astype(float) / 255.0
    
    return corrupted

def apply_anomalies(image_path, output_path, anomaly_types, intensity=0.5):
    """Apply specified anomalies to an image"""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] for processing
    image_norm = image.astype(float) / 255.0
    
    # Apply requested anomalies
    corrupted = image_norm.copy()
    
    if 'salt_pepper' in anomaly_types:
        corrupted = add_salt_pepper_noise(corrupted, amount=intensity/5)
    
    if 'gaussian' in anomaly_types:
        corrupted = add_gaussian_noise(corrupted, var=intensity/10)
    
    if 'speckle' in anomaly_types:
        corrupted = add_speckle_noise(corrupted, var=intensity/5)
    
    if 'poisson' in anomaly_types:
        corrupted = add_poisson_noise(corrupted)
    
    if 'local' in anomaly_types:
        corrupted = add_local_corruption(corrupted, intensity=intensity, 
                                         num_regions=int(10*intensity), 
                                         max_size=int(50*intensity))
    
    if 'drift' in anomaly_types:
        corrupted = add_pixel_drift(corrupted, intensity=int(20*intensity))
    
    if 'dead_pixels' in anomaly_types:
        corrupted = create_dead_pixels(corrupted, percentage=intensity/50)
    
    if 'bit_error' in anomaly_types:
        corrupted = add_bit_error(corrupted, bit_planes=int(3*intensity))
    
    # Display result
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_norm)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(corrupted, 0, 1))
    plt.title('Image with Anomalies')
    plt.axis('off')
    
    # Save result
    plt.tight_layout()
    plt.savefig(output_path)
    
    # Also save just the corrupted image
    corrupted_output = output_path.replace('.png', '_corrupted.png')
    cv2.imwrite(corrupted_output, cv2.cvtColor((corrupted*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    print(f"Original and corrupted images saved to {output_path}")
    print(f"Corrupted image saved to {corrupted_output}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Add pixel-level anomalies to an image')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--output', type=str, default='output.png', help='Path to save the output image')
    parser.add_argument('--intensity', type=float, default=0.5, help='Intensity of anomalies (0.0 to 1.0)')
    parser.add_argument('--anomalies', type=str, default='all', 
                        help='Comma-separated list of anomalies to apply: salt_pepper,gaussian,speckle,poisson,local,drift,dead_pixels,bit_error')
    
    args = parser.parse_args()
    
    # Parse anomaly types
    if args.anomalies.lower() == 'all':
        anomaly_types = ['salt_pepper', 'gaussian', 'speckle', 'poisson', 'local', 'drift', 'dead_pixels', 'bit_error']
    else:
        anomaly_types = [a.strip() for a in args.anomalies.split(',')]
    
    # Check intensity is in valid range
    intensity = max(0.0, min(1.0, args.intensity))
    
    # Apply anomalies
    apply_anomalies(args.image_path, args.output, anomaly_types, intensity)

if __name__ == "__main__":
    main()