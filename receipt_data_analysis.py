from flask import jsonify
import json

class ReceiptAnalyzer:
    def __init__(self, file_path="receipt_output.json"):
        self.file_path = file_path
        self.illegal_items = [
            "Cigarettes", "Cigars", "Vaping products", "Alcohol",
            "Recreational drugs", "Cannabis products", "Prescription medications (unauthorized)",
            "Mobile phones", "Tablets", "Smartwatches", "Personal laptops", "Gaming devices",
            "Headphones", "AirPods", "Bluetooth speakers", "Streaming subscriptions (Netflix, Spotify, etc.)",
            "Personal clothing", "Shoes", "Fashion accessories", "Luxury brand items",
            "Watches", "Designer bags", "Non-uniform business attire",
            "Movie tickets", "Concert passes", "Event passes", "Gym memberships",
            "Personal fitness equipment", "Spa treatments", "Beauty services",
            "Video games", "Gaming subscriptions",
            "Groceries for personal use", "Restaurant bills (non-work related)",
            "Energy drinks", "Dietary supplements",
            "First-class travel upgrades", "Luxury travel expenses", "Personal vehicle repairs",
            "Car maintenance", "Fuel expenses (non-business travel)", "Parking fines",
            "Traffic violations",
            "Furniture (unless approved)", "Home utility bills (electricity, water, internet)",
            "Decorative items (paintings, plants, etc.)",
            "Gifts for colleagues (unauthorized)", "Charitable donations",
            "Fines", "Penalties", "Legal fees", "Cryptocurrency transactions"
        ]
        
    def load_receipts(self):
        try:
            with open(self.file_path, 'r') as file:
                data = json.load(file)
                if isinstance(data, dict):
                    return [data]
                elif isinstance(data, list):
                    return data if all(isinstance(item, dict) for item in data) else []
                return []
        except (FileNotFoundError, json.JSONDecodeError):
            return []
            
    def fix_unit_prices(self, receipt_data):
        fixed_receipt = receipt_data.copy()
        if "Line Items" in fixed_receipt:
            for item in fixed_receipt["Line Items"]:
                quantity = float(item.get("Quantity", 1))
                total = float(item.get("Total", 0))
                if quantity > 0:  # Avoid division by zero
                    item["UnitPrice"] = round(total / quantity, 2)
                else:
                    item["UnitPrice"] = total
        return fixed_receipt
    
    def analyze_receipt(self, receipt_data):
    # Initialize results with proper structure matching frontend expectations
        results = {
            "illegal_items_found": [],
            "inconsistencies": [],
            "math_verification": {
                "calculated_subtotal": 0.0,
                "reported_subtotal": 0.0,
                "calculated_total": 0.0,
                "reported_total": 0.0
            },
            "is_math_correct": False
        }
        
        # Check for illegal items - store just the descriptions to match frontend
        if "Line Items" in receipt_data:
            for item in receipt_data["Line Items"]:
                desc = item.get("Description", "").lower()
                for illegal in self.illegal_items:
                    if illegal.lower() in desc:
                        # Just append description as per frontend code
                        results["illegal_items_found"].append(item["Description"])
                        # Add to inconsistencies
                        results["inconsistencies"].append(
                            f"Prohibited item detected: {item['Description']}"
                        )
        
        # Calculate and verify math
        subtotal = 0.0
        if "Line Items" in receipt_data:
            subtotal = sum(float(item.get("Total", 0)) for item in receipt_data["Line Items"])
        
        # Convert everything to float to ensure consistent types
        reported_subtotal = float(receipt_data.get("Subtotal", 0))
        tax = float(receipt_data.get("Tax Amount", 0))
        reported_total = float(receipt_data.get("Total Amount", 0))
        calculated_total = subtotal + tax
        
        # Update math verification
        results["math_verification"] = {
            "calculated_subtotal": subtotal,
            "reported_subtotal": reported_subtotal,
            "calculated_total": calculated_total,
            "reported_total": reported_total
        }
        
        # Check math accuracy (with small margin for floating point issues)
        is_subtotal_correct = abs(subtotal - reported_subtotal) < 0.01
        is_total_correct = abs(calculated_total - reported_total) < 0.01
        results["is_math_correct"] = is_subtotal_correct and is_total_correct
        
        # Add inconsistencies for math errors
        if not is_subtotal_correct:
            results["inconsistencies"].append(
                f"Subtotal calculation mismatch: reported ${reported_subtotal:.2f}, calculated ${subtotal:.2f}"
            )
        if not is_total_correct:
            results["inconsistencies"].append(
                f"Total calculation mismatch: reported ${reported_total:.2f}, calculated ${calculated_total:.2f}"
            )
            
        return results

if __name__ == "__main__":
    analyzer = ReceiptAnalyzer("uploads/processed_image_1743681717.json")
    receipts = analyzer.load_receipts()
    if receipts:
        for receipt in receipts:
            fixed_receipt = analyzer.fix_unit_prices(receipt)
            analysis_results = analyzer.analyze_receipt(fixed_receipt)
            print(json.dumps(analysis_results, indent=4))
            print(analysis_results["math_verification"])
            print(analysis_results["math_verification"]["calculated_subtotal"])
    
