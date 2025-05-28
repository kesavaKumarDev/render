import cv2
import numpy as np
import pytesseract
from PIL import Image

def load_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def resize_image(image, width=1000):
    aspect_ratio = float(width) / image.shape[1]
    new_height = int(image.shape[0] * aspect_ratio)
    resized = cv2.resize(image, (width, new_height))
    return resized

def denoise_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)  # Adjust kernel size as needed

def binarize_image(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

def remove_noise(image):
    kernel = np.ones((1, 1), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

def remove_background(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text(image):
    custom_config = r'--oem 3 --psm 6'  # Best for structured text
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

def preprocess_image(image_path):
    # Load and resize
    image, gray = load_image(image_path)
    resized = resize_image(gray)
    
    # Basic preprocessing
    denoised = denoise_image(resized)
    binary = binarize_image(denoised)
    cleaned = remove_noise(binary)
    background_removed = remove_background(cleaned)
    
    return background_removed


# Main processing
def process_receipt(image_path, output_path=None):
    # Preprocess image
    processed_img = preprocess_image(image_path)
    
    # Convert to PIL image
    result_image = Image.fromarray(processed_img)
    
    # Display the image
    result_image.show()
    
    # Save if output path provided
    if output_path:
        result_image.save(output_path)
    
    print("Receipt processed.")
    
    # Extract text
    text = extract_text(processed_img)
    print("\nExtracted Text:")
    print(text)
    
    return result_image, text


if __name__ == "__main__":
    image_path = "large-receipt-image-dataset-SRD/1010-receipt.jpg"
    output_path = "images/processed_receipt.jpg"
    
    processed_image, text = process_receipt(image_path, output_path)