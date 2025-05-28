# from PIL import Image
# from PIL.ExifTags import TAGS

# def extract_exif_data(image_path):
#     """ Extract EXIF metadata from an image and detect anomalies. """
    
#     try:
#         image = Image.open(image_path)
#         exif_data = image._getexif()  # Extract EXIF metadata

#         if not exif_data:
#             return "No EXIF metadata found. The image may have been edited or stripped."

#         metadata = {}
#         for tag_id, value in exif_data.items():
#             tag_name = TAGS.get(tag_id, tag_id)  # Convert tag ID to readable name
#             metadata[tag_name] = value

#         return metadata

#     except Exception as e:
#         return f"Error extracting EXIF data: {e}"

# # Example usage
# image_path = "Untitled design.jpg" 
# metadata = extract_exif_data(image_path)

# # ✅ FIX: Check if metadata is a dictionary before iterating
# if isinstance(metadata, dict):
#     for key, value in metadata.items():
#         print(f"{key}: {value}")
# else:
#     print(metadata)  # Print error message


# """
#      Missing EXIF Data → Some apps strip metadata to hide modifications.
#     Edited with Software → If "Software" = "Adobe Photoshop", it might be edited.
#     Timestamp Mismatch → If a receipt date ≠ EXIF timestamp, it could be fake.
#     Different Camera Model → If an invoice is claimed to be printed but shows a mobile camera EXIF, it 
# """


import os
import hashlib
from PIL import Image
import piexif
from datetime import datetime

def get_image_metadata(image_path):
    """Extract metadata from an image file"""
    img = Image.open(image_path)
    metadata = {}
    
    # Extract EXIF data
    exif_data = img.info.get("exif")
    if exif_data:
        exif_dict = piexif.load(exif_data)
        
        for ifd in ("0th", "Exif", "GPS", "1st"):
            for tag, value in exif_dict[ifd].items():
                tag_name = piexif.TAGS[ifd].get(tag, {}).get("name", tag)
                metadata[tag_name] = value
    
    return metadata

def check_image_modification(metadata):
    """Analyze metadata for possible modifications"""
    suspicious_keys = ["DateTime", "DateTimeOriginal", "DateTimeDigitized"]
    
    if not metadata:
        return "No metadata found, possible tampering."

    date_values = {}
    for key in suspicious_keys:
        if key in metadata:
            try:
                date_values[key] = datetime.strptime(metadata[key].decode(), "%Y:%m:%d %H:%M:%S")
            except (ValueError, AttributeError):
                return "Metadata timestamp error, possible tampering."

    if "DateTimeOriginal" in date_values and "DateTime" in date_values:
        if date_values["DateTime"] > date_values["DateTimeOriginal"]:
            return f"Image modified. Original: {date_values['DateTimeOriginal']}, Modified: {date_values['DateTime']}"
    
    return "No significant modifications detected."

def get_file_hash(file_path):
    """Generate hash for image integrity check"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def analyze_image(file_path):
    """Analyze image for metadata and modifications"""
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension not in [".jpg", ".jpeg", ".png"]:
        return "Unsupported file format."
    
    metadata = get_image_metadata(file_path)
    modification_status = check_image_modification(metadata)
    file_hash = get_file_hash(file_path)

    return {
        "metadata": metadata,
        "modification_status": modification_status,
        "file_hash": file_hash
    }

# Example Usage
file_path = "../Untitled design.jpg" # your image file
result = analyze_image(file_path)
print(result)


"""
✔ Extracts EXIF metadata (Date, Camera Info, GPS, etc.)
✔ Checks for modifications (Timestamp mismatches)
✔ Generates MD5 hash to detect changes
✔ Works with JPEG & PNG images
"""