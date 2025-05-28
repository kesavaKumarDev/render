import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from tqdm import tqdm
import re

class ReceiptTamperingGenerator:
    def __init__(self, 
                 input_dir, 
                 output_dir, 
                 num_tamperings_per_image=3, 
                 tampering_probability=0.8):
        """
        Initialize the receipt tampering generator.
        
        Args:
            input_dir (str): Directory containing original receipt images
            output_dir (str): Directory to save tampered receipt images
            num_tamperings_per_image (int): Number of tampered versions to create per original image
            tampering_probability (float): Probability of applying each tampering method
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_tamperings_per_image = num_tamperings_per_image
        self.tampering_probability = tampering_probability
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a directory for metadata
        self.metadata_dir = os.path.join(output_dir, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Initialize font for text manipulation
        try:
            self.font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            # Fallback to default font
            self.font = ImageFont.load_default()
        
        # Dictionary to store tampering metadata
        self.tampering_metadata = []
    
    def load_image(self, image_path):
        """Load an image from the specified path."""
        return Image.open(image_path).convert("RGB")
    
    def save_image(self, image, filename):
        """Save the image to the output directory."""
        save_path = os.path.join(self.output_dir, filename)
        image.save(save_path)
        return save_path
    
    def extract_text_regions(self, image):
        """
        Extract potential text regions from receipt image.
        This is a simplified version - in a real scenario, you'd use OCR.
        
        Returns:
            List of tuples (x, y, width, height) representing text regions
        """
        # Convert PIL image to OpenCV format
        img = np.array(image)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Use edge detection and dilation to find potential text regions
        edges = cv2.Canny(img_gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours of potential text regions
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on size
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter small regions that are likely to be text
            if 10 < w < 200 and 5 < h < 50:
                text_regions.append((x, y, w, h))
        
        return text_regions
    
    def extract_price_regions(self, image, text_regions):
        """
        Extract regions likely to contain prices.
        This is a simplified approach - in a real scenario you'd use OCR and pattern matching.
        
        Returns:
            List of tuples (x, y, width, height) representing potential price regions
        """
        # For simplicity, we'll assume price text regions are typically on the right side
        # of the receipt and have certain dimensions
        img_width = image.width
        price_regions = []
        
        for x, y, w, h in text_regions:
            # Prices are usually on the right side and have a specific shape
            if x > img_width * 0.5 and 10 < w < 100 and 5 < h < 30:
                price_regions.append((x, y, w, h))
        
        return price_regions
    
    def tamper_price(self, image, price_regions):
        """
        Tamper with price values in the receipt.
        
        Returns:
            Tampered image and description of tampering
        """
        if not price_regions:
            return image, "No price tampering performed"
        
        # Select a random price region to tamper with
        x, y, w, h = random.choice(price_regions)
        
        # Create a drawing context
        draw = ImageDraw.Draw(image)
        
        # Create a white rectangle to cover the original price
        draw.rectangle([(x, y), (x + w, y + h)], fill=(255, 255, 255))
        
        # Generate a new fake price
        new_price = f"${random.randint(1, 999)}.{random.randint(0, 99):02d}"
        
        # Draw the new price
        draw.text((x + 5, y + h//4), new_price, fill=(0, 0, 0), font=self.font)
        
        return image, f"Price changed at position ({x}, {y})"
    
    def tamper_date(self, image, text_regions):
        """
        Tamper with the date on the receipt.
        
        Returns:
            Tampered image and description of tampering
        """
        potential_date_regions = []
        
        # Dates are usually at the top of the receipt
        for x, y, w, h in text_regions:
            if y < image.height * 0.3 and 30 < w < 150:
                potential_date_regions.append((x, y, w, h))
        
        if not potential_date_regions:
            return image, "No date tampering performed"
        
        # Select a random date region to tamper with
        x, y, w, h = random.choice(potential_date_regions)
        
        # Create a drawing context
        draw = ImageDraw.Draw(image)
        
        # Create a white rectangle to cover the original date
        draw.rectangle([(x, y), (x + w, y + h)], fill=(255, 255, 255))
        
        # Generate a new fake date
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        year = random.randint(2020, 2025)
        new_date = f"{month:02d}/{day:02d}/{year}"
        
        # Draw the new date
        draw.text((x + 5, y + h//4), new_date, fill=(0, 0, 0), font=self.font)
        
        return image, f"Date changed at position ({x}, {y}) to {new_date}"
    
    def add_fake_item(self, image, text_regions):
        """
        Add a fake item to the receipt.
        
        Returns:
            Tampered image and description of tampering
        """
        item_regions = []
        
        # Items are usually in the middle section of the receipt
        for x, y, w, h in text_regions:
            if 0.2 * image.height < y < 0.8 * image.height and w > 50:
                item_regions.append((x, y, w, h))
        
        if not item_regions:
            return image, "No item addition performed"
        
        # Find a space to insert the new item (after an existing item)
        if item_regions:
            region = random.choice(item_regions)
            x, y, w, h = region
            insert_y = y + h + 2  # Insert just below the selected item
        else:
            # Fallback if no suitable regions found
            x = int(image.width * 0.1)
            insert_y = int(image.height * 0.5)
            w = int(image.width * 0.6)
            h = 20
        
        # Create fake items
        fake_items = [
            "Coffee",
            "Bottled Water",
            "Snack",
            "Sandwich",
            "Salad",
            "Dessert",
            "Pizza",
            "Burger",
            "Drink",
            "Fries"
        ]
        fake_item = random.choice(fake_items)
        fake_price = f"${random.randint(1, 30)}.{random.randint(0, 99):02d}"
        
        # Create a drawing context
        draw = ImageDraw.Draw(image)
        
        # Draw a white background for the new item
        background_height = h + 4
        draw.rectangle([(x, insert_y), (x + w + 100, insert_y + background_height)], 
                      fill=(255, 255, 255))
        
        # Draw the fake item text
        draw.text((x + 5, insert_y + 2), fake_item, fill=(0, 0, 0), font=self.font)
        
        # Draw the fake price to the right
        price_x = x + w - 30
        draw.text((price_x, insert_y + 2), fake_price, fill=(0, 0, 0), font=self.font)
        
        return image, f"Added fake item '{fake_item}' for {fake_price} at position ({x}, {insert_y})"
    
    def modify_total(self, image, text_regions):
        """
        Modify the total amount on the receipt.
        
        Returns:
            Tampered image and description of tampering
        """
        potential_total_regions = []
        
        # Totals are usually at the bottom of the receipt
        for x, y, w, h in text_regions:
            if y > image.height * 0.6 and x > image.width * 0.5:
                potential_total_regions.append((x, y, w, h))
        
        if not potential_total_regions:
            return image, "No total modification performed"
        
        # Select a random total region to tamper with
        x, y, w, h = random.choice(potential_total_regions)
        
        # Create a drawing context
        draw = ImageDraw.Draw(image)
        
        # Create a white rectangle to cover the original total
        draw.rectangle([(x, y), (x + w, y + h)], fill=(255, 255, 255))
        
        # Generate a new fake total
        new_total = f"${random.randint(10, 500)}.{random.randint(0, 99):02d}"
        
        # Draw the new total
        draw.text((x + 5, y + h//4), new_total, fill=(0, 0, 0), font=self.font)
        
        return image, f"Total modified at position ({x}, {y}) to {new_total}"
    
    def simulate_erasure(self, image, text_regions):
        """
        Simulate erasure on the receipt.
        
        Returns:
            Tampered image and description of tampering
        """
        if not text_regions:
            return image, "No erasure simulation performed"
        
        # Select a random text region to erase
        x, y, w, h = random.choice(text_regions)
        
        # Create a drawing context
        draw = ImageDraw.Draw(image)
        
        # Create a light smudge effect
        for i in range(20):
            smudge_x = x + random.randint(0, w)
            smudge_y = y + random.randint(0, h)
            smudge_size = random.randint(3, 10)
            smudge_opacity = random.randint(200, 255)
            draw.ellipse([(smudge_x, smudge_y), 
                         (smudge_x + smudge_size, smudge_y + smudge_size)], 
                        fill=(smudge_opacity, smudge_opacity, smudge_opacity))
        
        return image, f"Erasure simulated at position ({x}, {y})"
    
    def add_noise(self, image):
        """
        Add random noise to the image to simulate scanning artifacts.
        
        Returns:
            Tampered image and description of tampering
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Add random noise
        noise = np.random.normal(0, 15, img_array.shape).astype(np.uint8)
        noisy_img = cv2.add(img_array, noise)
        
        # Convert back to PIL
        noisy_image = Image.fromarray(noisy_img)
        
        return noisy_image, "Added random noise to simulate scanning artifacts"
    
    def simulate_fold(self, image):
        """
        Simulate a fold mark across the receipt.
        
        Returns:
            Tampered image and description of tampering
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Decide orientation of fold (horizontal or vertical)
        if random.random() > 0.5:
            # Horizontal fold
            fold_y = random.randint(int(image.height * 0.2), int(image.height * 0.8))
            fold_width = random.randint(5, 15)
            
            # Create a lighter region for the fold
            fold_effect = np.ones((fold_width, image.width, 3)) * 30
            img_array[fold_y:fold_y+fold_width, :] = cv2.add(
                img_array[fold_y:fold_y+fold_width, :], 
                fold_effect.astype(np.uint8)
            )
            
            description = f"Horizontal fold simulated at y={fold_y}"
        else:
            # Vertical fold
            fold_x = random.randint(int(image.width * 0.2), int(image.width * 0.8))
            fold_width = random.randint(5, 15)
            
            # Create a lighter region for the fold
            fold_effect = np.ones((image.height, fold_width, 3)) * 30
            img_array[:, fold_x:fold_x+fold_width] = cv2.add(
                img_array[:, fold_x:fold_x+fold_width], 
                fold_effect.astype(np.uint8)
            )
            
            description = f"Vertical fold simulated at x={fold_x}"
        
        # Convert back to PIL
        folded_image = Image.fromarray(img_array)
        
        return folded_image, description
    
    def create_tampered_receipt(self, image_path):
        """
        Create a tampered version of a receipt image by applying various tampering methods.
        
        Args:
            image_path (str): Path to the original receipt image
            
        Returns:
            str: Path to the saved tampered image
            dict: Metadata about the tampering
        """
        # Load the image
        image = self.load_image(image_path)
        
        # Extract text regions
        text_regions = self.extract_text_regions(image)
        
        # Extract price regions
        price_regions = self.extract_price_regions(image, text_regions)
        
        # List of tampering methods
        tampering_methods = [
            (self.tamper_price, [image, price_regions]),
            (self.tamper_date, [image, text_regions]),
            (self.add_fake_item, [image, text_regions]),
            (self.modify_total, [image, text_regions]),
            (self.simulate_erasure, [image, text_regions]),
            (self.add_noise, [image]),
            (self.simulate_fold, [image])
        ]
        
        # Select and apply tampering methods based on probability
        applied_tamperings = []
        tampered_image = image.copy()
        
        for method, args in tampering_methods:
            if random.random() < self.tampering_probability:
                updated_args = [tampered_image] + args[1:]
                tampered_image, description = method(*updated_args)
                applied_tamperings.append(description)
        
        # Generate a filename for the tampered image
        base_filename = os.path.basename(image_path)
        name, ext = os.path.splitext(base_filename)
        tampered_filename = f"{name}_tampered_{len(self.tampering_metadata)}{ext}"
        
        # Save the tampered image
        saved_path = self.save_image(tampered_image, tampered_filename)
        
        # Create metadata
        metadata = {
            "original_image": image_path,
            "tampered_image": saved_path,
            "tampering_methods": applied_tamperings,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Add to tampering metadata list
        self.tampering_metadata.append(metadata)
        
        return saved_path, metadata
    
    def generate_tampered_dataset(self):
        """
        Process all images in the input directory and create tampered versions.
        """
        # Get list of image files
        image_files = [f for f in os.listdir(self.input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        print(f"Found {len(image_files)} receipt images in the input directory.")
        
        # Process each image
        for image_file in tqdm(image_files, desc="Generating tampered receipts"):
            image_path = os.path.join(self.input_dir, image_file)
            
            # Create multiple tampered versions per original image
            for i in range(self.num_tamperings_per_image):
                try:
                    _, _ = self.create_tampered_receipt(image_path)
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
        
        # Save metadata to CSV
        self.save_metadata()
        
        print(f"Generated {len(self.tampering_metadata)} tampered receipt images.")
    
    def save_metadata(self):
        """Save the tampering metadata to a CSV file."""
        metadata_df = pd.DataFrame(self.tampering_metadata)
        metadata_path = os.path.join(self.metadata_dir, "tampering_metadata.csv")
        metadata_df.to_csv(metadata_path, index=False)
        
        # Also save a simplified version with one row per tampering method
        tampering_details = []
        
        for item in self.tampering_metadata:
            original_img = item['original_image']
            tampered_img = item['tampered_image']
            
            for method in item['tampering_methods']:
                tampering_details.append({
                    'original_image': original_img,
                    'tampered_image': tampered_img,
                    'tampering_method': method
                })
        
        details_df = pd.DataFrame(tampering_details)
        details_path = os.path.join(self.metadata_dir, "tampering_details.csv")
        details_df.to_csv(details_path, index=False)

# Example usage
if __name__ == "__main__":
    # Set directories
    input_dir = "./large-receipt-image-dataset-SRD"
    output_dir = "./tampered-images"
    
    # Create tampering generator
    generator = ReceiptTamperingGenerator(
        input_dir=input_dir,
        output_dir=output_dir,
        num_tamperings_per_image=3,
        tampering_probability=0.7
    )
    
    # Generate the tampered dataset
    generator.generate_tampered_dataset()