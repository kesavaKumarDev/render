from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
import time
import io
import json
import numpy as np

# MongoDB imports
from pymongo import MongoClient
from bson.objectid import ObjectId
import gridfs

# Image processing imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Custom modules
from ela_analyzer import ErrorLevelAnalyzer
# from deepfake_detector import DeepFakeDetector
from clone_detection import CloneDetector
from Forgery import ForgeryDetector
from pixel_level_anomaly import ReceiptTamperingDetector
from receipt_parser import ReceiptParser
from receipt_data_analysis import ReceiptAnalyzer
app = Flask(__name__)
CORS(app)

# MongoDB Configuration
client = MongoClient('mongodb://localhost:27017/')
db = client['image_analysis_db']
fs = gridfs.GridFS(db)

# Collections
analyses_collection = db['analyses']
text_analyses_collection = db['text_analyses']


# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/save-analysis', methods=['POST'])
def save_analysis():
    try:
        # Get the image file
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No image selected'}), 400
        
        # Get analysis data
        analysis_type = request.form.get('analysisType', '')
        analysis_subtype = request.form.get('analysisSubtype', '')
        analysis_result = request.form.get('analysisResult', '{}')
        
        # Parse analysis result to ensure it's valid JSON
        try:
            analysis_result_json = json.loads(analysis_result)
        except json.JSONDecodeError:
            analysis_result_json = {}
        
        # Generate unique ID for the original image
        original_image_id = str(uuid.uuid4())
        
        # Store the original image in GridFS
        image_file_id = fs.put(
            file.read(), 
            filename=secure_filename(file.filename),
            content_type=file.content_type
        )
        
        # Store result image in GridFS if provided
        result_image_file_id = None
        if 'resultImage' in request.files:
            result_image = request.files['resultImage']
            if result_image.filename != '':
                result_image_file_id = fs.put(
                    result_image.read(),
                    filename=secure_filename(result_image.filename),
                    content_type=result_image.content_type
                )
        
        # Prepare document for MongoDB
        document = {
            'originalImageId': str(image_file_id),
            'resultImageId': str(result_image_file_id) if result_image_file_id else None,
            'analysisType': analysis_type,
            'analysisSubtype': analysis_subtype,
            'fileName': secure_filename(file.filename),
            'fileSize': request.form.get('fileSize', '0'),
            'analysisResult': analysis_result_json,
            'createdAt': datetime.now()
        }
        
        # Insert into MongoDB
        result = analyses_collection.insert_one(document)
        
        return jsonify({
            'success': True, 
            'message': 'Analysis saved successfully', 
            'analysisId': str(result.inserted_id)
        }), 201
    
    except Exception as e:
        print(f"Error saving analysis: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/api/get-analyses', methods=['GET'])
def get_analyses():
    try:
        # Get all analyses, sorted by creation date (newest first)
        analyses = list(analyses_collection.find().sort('createdAt', -1))
        
        # Convert ObjectId to string for JSON serialization
        for analysis in analyses:
            analysis['_id'] = str(analysis['_id'])
            analysis['createdAt'] = analysis['createdAt'].isoformat()
        
        return jsonify({
            'success': True,
            'analyses': analyses
        })
    
    except Exception as e:
        print(f"Error retrieving analyses: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route("/api/delete-all-analyses", methods=["DELETE"])
def delete_all_analyses():
    try:
        # Delete all analyses from your MongoDB collection
        db.analyses.delete_many({})
        # Also delete associated files/images if needed
        
        return jsonify({"success": True, "message": "All analyses deleted successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/get-analysis/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    try:
        # Find analysis by ID
        analysis = analyses_collection.find_one({'_id': ObjectId(analysis_id)})
        
        if not analysis:
            return jsonify({'success': False, 'message': 'Analysis not found'}), 404
        
        # Convert ObjectId to string for JSON serialization
        analysis['_id'] = str(analysis['_id'])
        analysis['createdAt'] = analysis['createdAt'].isoformat()
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
    
    except Exception as e:
        print(f"Error retrieving analysis: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/api/get-image/<image_id>', methods=['GET'])
def get_image(image_id):
    try:
        # Find and retrieve the image from GridFS
        image = fs.get(ObjectId(image_id))
        
        if not image:
            return jsonify({'success': False, 'message': 'Image not found'}), 404
        
        # Create a temporary file path
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{image.filename}")
        
        # Write the image data to the temporary file
        with open(temp_file_path, 'wb') as f:
            f.write(image.read())
        
        # Send the file
        return send_file(temp_file_path, mimetype=image.content_type)
    
    except Exception as e:
        print(f"Error retrieving image: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/api/delete-analysis/<analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    try:
        # Find the analysis first to get file IDs
        analysis = analyses_collection.find_one({'_id': ObjectId(analysis_id)})
        
        if not analysis:
            return jsonify({'success': False, 'message': 'Analysis not found'}), 404
        
        # Delete the original image from GridFS
        if 'originalImageId' in analysis and analysis['originalImageId']:
            fs.delete(ObjectId(analysis['originalImageId']))
        
        # Delete result image if it exists
        if 'resultImageId' in analysis and analysis['resultImageId']:
            fs.delete(ObjectId(analysis['resultImageId']))
        
        # Delete the analysis document
        analyses_collection.delete_one({'_id': ObjectId(analysis_id)})
        
        return jsonify({
            'success': True,
            'message': 'Analysis deleted successfully'
        })
    
    except Exception as e:
        print(f"Error deleting analysis: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

# Add this to your existing analysis endpoints to save results automatically
def save_analysis_result(original_image, analysis_type, analysis_subtype, result_data, result_image=None):
    try:
        # Store the original image in GridFS
        original_image_id = fs.put(
            original_image.read(), 
            filename=secure_filename(original_image.filename),
            content_type=original_image.content_type
        )
        
        # Store result image in GridFS if provided
        result_image_id = None
        if result_image:
            result_image_id = fs.put(
                result_image,
                filename=f"result_{secure_filename(original_image.filename)}",
                content_type='image/png'  # Adjust content type as needed
            )
        
        # Prepare document for MongoDB
        document = {
            'originalImageId': str(original_image_id),
            'resultImageId': str(result_image_id) if result_image_id else None,
            'analysisType': analysis_type,
            'analysisSubtype': analysis_subtype,
            'fileName': secure_filename(original_image.filename),
            'fileSize': len(original_image.read()),
            'analysisResult': result_data,
            'createdAt': datetime.now()
        }
        
        # Reset file pointer
        original_image.seek(0)
        
        # Insert into MongoDB
        analyses_collection.insert_one(document)
        
        return True
    
    except Exception as e:
        print(f"Error automatically saving analysis: {str(e)}")
        return False




# Initialize detectors
# deepfake_detector = DeepFakeDetector()
forgery_detector = ForgeryDetector()
clone_detector = CloneDetector()
analyzer = ReceiptAnalyzer()
parser = ReceiptParser()

@app.route('/api/analyze-and-visualize', methods=['POST'])
def analyze_and_visualize():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    quality = int(request.form.get('quality', 90))
    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    image_file.save(filepath)

    try:
        analyzer = ErrorLevelAnalyzer(quality=quality)
        results = analyzer.get_analysis_results(filepath)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(np.array(results["original_image"]))
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(np.array(results["ela_image"]), cmap="gray")
        axes[1].set_title(f"Error Level Analysis (ELA)\nSuspicious Regions: {results['suspicious_regions_count']}")
        axes[1].axis("off")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)

        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/api/clone-detection', methods=['POST'])
def analyze_forgery():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(filepath)

    try:
        result_img, message = clone_detector.detect_forgery_sift(filepath)

        if result_img is None:
            return jsonify({"error": message}), 400

        # Convert BGR to RGB if necessary
        if len(result_img.shape) == 3 and result_img.shape[2] == 3:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        buf = io.BytesIO()
        plt.figure()  # Create a new figure
        plt.imshow(result_img, cmap='gray' if len(result_img.shape) == 2 else None)
        plt.title("Copy-Move Forgery Detection (SIFT Feature Matching)")
        plt.axis('off')
        plt.savefig(buf, format='png')
        plt.close()  # Close the figure
        buf.seek(0)

        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# @app.route("/api/deepfake-detection", methods=["POST"])
# def detect_deepfake():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     image_file = request.files["image"]
#     image_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_image.jpg")
#     image_file.save(image_path)

#     try:
#         result = deepfake_detector.predict(image_path)
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#     finally:
#         if os.path.exists(image_path):
#             os.remove(image_path)



@app.route('/api/anomaly-detection', methods=['POST'])
def detect_anomalies():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    try:
        anomaly_detector = ReceiptTamperingDetector(image_path)
        results = anomaly_detector.detect_anomalies()
        summary = anomaly_detector.summarize_anomalies()
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


@app.route("/api/forgery-detection", methods=["POST"])
def detect_tampering():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp.jpg")
    file.save(image_path)

    try:
        result = forgery_detector.detect_tampering(image_path)

        # Convert NumPy booleans to Python booleans
        result = {key: bool(value) if isinstance(value, np.bool_) else value for key, value in result.items()}

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


@app.route('/api/analyze-receipt', methods=['POST'])
def analyze_single_receipt():
    try:
        # If a filename is provided, use that file
        if request.json and 'filename' in request.json:
            json_filename = request.json.get('filename')
            json_file_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
           
            if not os.path.exists(json_file_path):
                return jsonify({"error": f"Receipt file {json_filename} not found"}), 404
               
            with open(json_file_path, 'r') as f:
                receipt_data = json.load(f)
        # Otherwise use the direct JSON data
        elif request.json:
            receipt_data = request.json
        else:
            return jsonify({"error": "No receipt data or filename provided"}), 400
           
        if not isinstance(receipt_data, dict):
            return jsonify({"error": "Invalid receipt format"}), 400
           
        # Create an instance of ReceiptAnalyzer if not already created
        analyzer = ReceiptAnalyzer()
        
        # First fix the unit prices in the receipt data
        fixed_receipt = analyzer.fix_unit_prices(receipt_data)
        
        # Then analyze the fixed receipt data
        result = analyzer.analyze_receipt(fixed_receipt)
        print(result)
        
        return jsonify(result)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON file format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/parse-receipt', methods=['POST'])
def parse_receipt():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
   
    file = request.files['image']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    try:
        # Create an instance of ReceiptParser
        parser = ReceiptParser()
        
        # Parse the receipt image
        receipt_data = parser.analyze_receipt(file_path)
        
        if receipt_data is None:
            return jsonify({'error': 'Could not parse receipt. Try a clearer image.'}), 400
       
        # Generate a unique JSON filename
        json_filename = f"{os.path.splitext(file.filename)[0]}_{int(time.time())}.json"
        json_file_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
        
        # Save the receipt data as JSON
        with open(json_file_path, 'w') as f:
            json.dump(receipt_data, f, indent=4)
            
        # Standardize field names to work with the analyzer
        if 'line_items' in receipt_data and 'Line Items' not in receipt_data:
            receipt_data['Line Items'] = receipt_data['line_items']
        
        # Also calculate unit prices if needed
        analyzer = ReceiptAnalyzer()
        fixed_receipt = analyzer.fix_unit_prices(receipt_data)
        
        return jsonify({
            'data': fixed_receipt,
            'filename': json_filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the uploaded image file
        if os.path.exists(file_path):
            os.remove(file_path)

# Save text analysis results
@app.route('/api/save-text-analysis', methods=['POST'])
def save_text_analysis():
    try:
        # Check if image file was uploaded
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'}), 400
        
        # Save the original image to GridFS
        file_id = fs.put(
            file.read(),
            filename=secure_filename(file.filename),
            content_type=file.content_type
        )
        
        # Get form data
        extracted_text = request.form.get('extractedText', '')
        # parsed_data = json.loads(request.form.get('parsedData', '{}'))
        # In your save-text-analysis endpoint
        # Validate the parsed data before saving
        try:
            parsed_data = json.loads(request.form.get('parsedData', '{}'))
            if parsed_data:
                # Ensure numeric values are properly stored as numbers
                if 'Total Amount' in parsed_data and isinstance(parsed_data['Total Amount'], str):
                    parsed_data['Total Amount'] = float(parsed_data['Total Amount'])
                if 'Subtotal' in parsed_data and isinstance(parsed_data['Subtotal'], str):
                    parsed_data['Subtotal'] = float(parsed_data['Subtotal'])
                if 'Tax Amount' in parsed_data and isinstance(parsed_data['Tax Amount'], str):
                    parsed_data['Tax Amount'] = float(parsed_data['Tax Amount'])
        except Exception as e:
            print(f"Error parsing data: {str(e)}")
        analysis_results = json.loads(request.form.get('analysisResults', '{}'))
        file_size = request.form.get('fileSize', '0')
        file_name = request.form.get('fileName', 'unnamed_file')
        
        # Create a new document in the text_analyses collection
        text_analysis_doc = {
            'fileName': file_name,
            'fileSize': int(file_size),
            'originalImageId': file_id,
            'extractedText': extracted_text,
            'parsedData': parsed_data,
            'analysisResults': analysis_results,
            'createdAt': datetime.now()
        }
        
        result = text_analyses_collection.insert_one(text_analysis_doc)
        
        return jsonify({
            'success': True,
            'message': 'Text analysis saved successfully',
            'analysisId': str(result.inserted_id)
        })
        
    except Exception as e:
        print(f"Error saving text analysis: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

# Get all saved text analyses
@app.route('/api/get-text-analyses', methods=['GET'])
def get_text_analyses():
    try:
        analyses = []
        for doc in text_analyses_collection.find().sort('createdAt', -1):
            doc['_id'] = str(doc['_id'])
            if 'originalImageId' in doc:
                doc['originalImageId'] = str(doc['originalImageId'])
            analyses.append(doc)
        
        return jsonify({'success': True, 'analyses': analyses})
    except Exception as e:
        print(f"Error retrieving text analyses: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

# Get a specific text analysis by ID
@app.route('/api/get-text-analysis/<analysis_id>', methods=['GET'])
def get_text_analysis(analysis_id):
    try:
        doc = text_analyses_collection.find_one({'_id': ObjectId(analysis_id)})
        if not doc:
            return jsonify({'success': False, 'message': 'Analysis not found'}), 404
        
        doc['_id'] = str(doc['_id'])
        if 'originalImageId' in doc:
            doc['originalImageId'] = str(doc['originalImageId'])
        
        return jsonify({'success': True, 'analysis': doc})
    except Exception as e:
        print(f"Error retrieving text analysis: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

# Delete a specific text analysis by ID
@app.route('/api/delete-text-analysis/<analysis_id>', methods=['DELETE'])
def delete_text_analysis(analysis_id):
    try:
        # Find the analysis first to get its image ID
        doc = text_analyses_collection.find_one({'_id': ObjectId(analysis_id)})
        if not doc:
            return jsonify({'success': False, 'message': 'Analysis not found'}), 404
        
        # Delete the associated image from GridFS
        if 'originalImageId' in doc:
            try:
                fs.delete(ObjectId(doc['originalImageId']))
            except:
                # Continue even if image deletion fails
                pass
        
        # Delete the analysis document
        result = text_analyses_collection.delete_one({'_id': ObjectId(analysis_id)})
        
        if result.deleted_count > 0:
            return jsonify({'success': True, 'message': 'Analysis deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to delete analysis'}), 500
            
    except Exception as e:
        print(f"Error deleting text analysis: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

# Delete all text analyses
@app.route('/api/delete-all-text-analyses', methods=['DELETE'])
def delete_all_text_analyses():
    try:
        # Find all analyses to get their image IDs
        all_analyses = list(text_analyses_collection.find({}, {'originalImageId': 1}))
        
        # Delete all associated images from GridFS
        for doc in all_analyses:
            if 'originalImageId' in doc:
                try:
                    fs.delete(ObjectId(doc['originalImageId']))
                except:
                    # Continue even if image deletion fails
                    pass
        
        # Delete all analysis documents
        result = text_analyses_collection.delete_many({})
        
        return jsonify({
            'success': True, 
            'message': f'All text analyses deleted successfully. Total deleted: {result.deleted_count}'
        })
        
    except Exception as e:
        print(f"Error deleting all text analyses: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
