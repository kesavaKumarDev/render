import pytesseract
from PIL import Image
import ollama
import json
import re
import cv2
import numpy as np
class ReceiptParser:
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    def __init__(self, ollama_model='llama3'):
        """
        Initialize the Receipt Parser with Ollama model
        :param ollama_model: Name of the Ollama model to use (default: llama3)
        """
        self.ollama_model = ollama_model
    def assess_image_quality(self, image_path):
        """
        Assess the quality of the input image with more comprehensive checks
        :param image_path: Path to the receipt image
        :return: Dictionary with image quality metrics
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Calculate brightness
            brightness = np.mean(gray)
            # Calculate contrast using standard deviation
            contrast = np.std(gray)
            # Calculate blurriness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Check text region density
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text_density = np.sum(binary == 255) / binary.size
            # Check for skew/rotation
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            # Calculate line angle variation
            line_angles = []
            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                        line_angles.append(angle)
            angle_variation = np.std(line_angles) if line_angles else 0
            # Additional checks
            is_too_noisy = text_density < 0.05  # Very low text density
            is_severely_skewed = angle_variation > 60  # High angle variation suggests severe skew
            # Determine overall image quality
            quality_assessment = {
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': laplacian_var,
                'text_density': text_density,
                'angle_variation': angle_variation,
                'is_suitable': all([
                    50 < brightness < 300,  # Reasonable brightness range
                    contrast > 20,           # Minimum contrast
                    laplacian_var > 100,     # Minimum sharpness
                    0.1 < text_density < 1.0,  # Reasonable text density
                    angle_variation < 60,    # Minimal skew
                    not is_too_noisy,        # Not too noisy
                    not is_severely_skewed   # Not severely skewed
                ])
            }
            print("Image Quality Assessment:", quality_assessment)
            return quality_assessment
        except Exception as e:
            print(f"Error assessing image quality: {e}")
            return None
    def enhance_image(self, image_path):
        """
        Enhance image quality for better OCR
        :param image_path: Path to the receipt image
        :return: Path to the enhanced image
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            # Denoise
            denoised = cv2.fastNlMeansDenoising(adaptive_thresh, None, 10, 7, 21)
            # Enhanced image path
            enhanced_path = "enhanced_receipt.png"
            cv2.imwrite(enhanced_path, denoised)
            return enhanced_path
        except Exception as e:
            print(f"Error enhancing image: {e}")
            return image_path
    def extract_text_from_image(self, image_path):
        """
        Extract text from receipt image using Tesseract
        :param image_path: Path to the receipt image
        :return: Extracted text string
        """
        try:
            # First, assess image quality
            quality = self.assess_image_quality(image_path)
            if not quality or not quality['is_suitable']:
                print("Image quality is poor. Attempting to enhance...")
                image_path = self.enhance_image(image_path)
            # Read and process image
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            processed_image_path = "processed_image.png"
            cv2.imwrite(processed_image_path, gray)
            # Extract text
            raw_text = pytesseract.image_to_string(Image.open(processed_image_path))
            cleaned_text = self._preprocess_text(raw_text)
            print("Text extraction completed.")
            print(cleaned_text)
            return cleaned_text
        except Exception as e:
            print(f"Error extracting text: {e}")
            return None
    def _preprocess_text(self, text):
        """
        Preprocess the extracted text
        :param text: Raw extracted text
        :return: Cleaned and preprocessed text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable())
        return text
    def generate_structured_data(self, raw_text):
        """
        Use Ollama to convert raw text into structured JSON.
        Ensures line items have a default quantity of 1 if not specified.
        """
        prompt = f"""
    Parse the following receipt text and return a structured JSON with these key details:
        - Company/Shop/Restaurant Name
        - Address
        - Total Amount
        - Subtotal
        - Tax Amount
        - Line Items (with Quantity, Description, Unit Price, Total)
        - Date of Purchase
        - Payment Method (if available)

        IMPORTANT PARSING RULES:
        1. If quantity is not explicitly mentioned for a line item, use 1 as default
        2. When a quantity greater than 1 is specified:
        - The price shown on the receipt is the TOTAL for that line item
        - Calculate the Unit Price by dividing the total price by the quantity
        3. For line items with quantity of 1, the price shown is both the unit price and total

        Raw Receipt Text:
        {raw_text}

        Ensure the output is a **valid JSON dictionary**. If any field is missing, set it to `null`.
        Format numbers as decimal values (e.g., 12.00 not 12).
                """
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            raw_content = response.get('message', {}).get('content', '').strip()
            if not raw_content:
                raise ValueError("Ollama returned an empty response.")
            # Attempt to extract JSON by finding the first '{' and last '}'
            json_start = raw_content.find('{')
            json_end = raw_content.rfind('}')
            if json_start == -1 or json_end == -1:
                raise ValueError("No valid JSON found in Ollama response.")
            json_content = raw_content[json_start:json_end + 1]  # Extract JSON part
            structured_data = json.loads(json_content)  # Parse JSON
            # Ensure each line item has a quantity of 1 if not specified
            if 'line_items' in structured_data:
                for item in structured_data['line_items']:
                    if 'quantity' not in item or item['quantity'] is None:
                        item['quantity'] = 1
            return structured_data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Raw Ollama Response:\n{raw_content}")  # Debugging info
            return None
        except Exception as e:
            print(f"Error parsing receipt: {e}")
            return None
    def analyze_receipt(self, image_path):
        """
        Complete receipt analysis pipeline
        :param image_path: Path to receipt image
        :return: Structured receipt data or None
        """
        # Assess image quality first
        quality = self.assess_image_quality(image_path)
        # If image quality is not suitable, print a statement and return None
        if not quality or not quality['is_suitable']:
            print("Image quality is insufficient for receipt parsing. Please provide a clearer image.")
            return None
        # Extract text from image
        raw_text = self.extract_text_from_image(image_path)
        if not raw_text:
            return None
        # Generate structured data
        structured_data = self.generate_structured_data(raw_text)
        return structured_data
    def save_to_json(self, data, filename="receipt_output.json"):
        """
        Save structured receipt data to a JSON file.
        :param data: Parsed receipt data (dictionary)
        :param filename: Name of the output JSON file
        """
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            print(f"Receipt data saved to {filename}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")
# Example usage
def main():
    # Initialize parser
    parser = ReceiptParser()
    # Path to your receipt image
    image_path = './large-receipt-image-dataset-SRD/1007-receipt.jpg'
    # Assess and extract from receipt
    receipt_data = parser.analyze_receipt(image_path)
    if receipt_data:
        # Save structured data to JSON
        parser.save_to_json(receipt_data)
        # Print structured data
        print(json.dumps(receipt_data, indent=2))
if __name__ == "__main__":
    main()


