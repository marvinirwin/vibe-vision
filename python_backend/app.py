import os
import base64
import io
import random
import time
from functools import wraps
import traceback

import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from PIL import Image
import cv2 # OpenCV for image processing

# --- Model Imports (potentially time-consuming) ---
print("Loading models...")
try:
    from ultralytics import YOLO
    # Load YOLO models (adjust model names as needed, e.g., yolov8n.pt, yolov8n-pose.pt)
    # Using smaller models for quicker loading in demo
    yolo_entity_model = YOLO('yolov10n.pt')
    yolo_pose_model = YOLO('yolov8n-pose.pt')
    print("YOLO models loaded.")
except ImportError as e:
    print(f"Warning: Failed to import ultralytics. YOLO endpoints will not work. Error: {e}")
    YOLO = None
    yolo_entity_model = None
    yolo_pose_model = None

try:
    from deepface import DeepFace
    # Pre-load DeepFace models to avoid delay on first request
    # This builds the ArcFace model by default upon first use in analyze/represent
    # You can force a specific model build if needed, but analyze/represent usually handles it.
    # Example: DeepFace.build_model("ArcFace") 
    print("DeepFace imported. Models will be built on first use.")
except ImportError as e:
    print(f"Warning: Failed to import deepface. ArcFace endpoints will not work. Error: {e}")
    DeepFace = None

print("Model loading complete.")
# ----------------------------

# --- Configuration & Setup ---
load_dotenv()  # Load environment variables from .env file
app = Flask(__name__)

# --- Gemini Configuration ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MODEL_NAME = "gemini-1.5-flash"
genai_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        genai_model = genai.GenerativeModel(
            MODEL_NAME,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
        )
        print("Gemini Model Initialized Successfully.")
    except Exception as e:
        print(f"Error initializing Gemini Model: {e}")
else:
    print("Gemini model not initialized due to missing API key.")

# --- Helper Functions ---
def require_image_data(f):
    """Decorator to check for imageData in request JSON."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return jsonify({"message": "Request must be JSON"}), 400
        data = request.get_json()
        if 'imageData' not in data:
            return jsonify({"message": "Missing image data"}), 400
        kwargs['image_data_b64'] = data['imageData']
        kwargs['request_data'] = data # Pass full data for functions needing more args
        return f(*args, **kwargs)
    return decorated_function

def base64_to_cv2(base64_string):
    """Converts a base64 image string to an OpenCV image (NumPy array)."""
    try:
        if ',' in base64_string:
            header, encoded = base64_string.split(',', 1)
        else:
            encoded = base64_string
        
        img_data = base64.b64decode(encoded)
        img_np = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if image is None:
             raise ValueError("cv2.imdecode returned None")
        return image
    except Exception as e:
        print(f"Error decoding base64 to cv2 image: {e}")
        traceback.print_exc()
        return None

def base64_to_gemini_part(base64_string):
    """Converts base64 string to Gemini Part object."""
    try:
        # Use regex to handle potential prefixes like data:image/jpeg;base64,
        import re # Import re here if not already imported globally
        match = re.match(r'^data:(image/\w+);base64,(.*)', base64_string)
        if match:
            mime_type = match.group(1)
            data = match.group(2)
            return {"mime_type": mime_type, "data": data}
        else:
            # Attempt to guess common types if no header (less reliable)
            try:
                image_bytes = base64.b64decode(base64_string)
                img = Image.open(io.BytesIO(image_bytes))
                mime_type = Image.MIME.get(img.format)
                if mime_type:
                    # Need to re-encode the *original* string data part for Gemini
                    return {"mime_type": mime_type, "data": base64_string}
                else:
                    print("Could not determine MIME type from base64 without header.")
                    return None
            except Exception as inner_e:
                 print(f"Error guessing MIME type: {inner_e}")
                 return None
    except Exception as e:
        print(f"Error processing base64 for Gemini: {e}")
        return None

# --- API Routes ---

@app.route('/')
def health_check():
    return "Vibe Vision Python Backend is running!"

@app.route('/api/yolo/detect-entities', methods=['POST'])
@require_image_data
def detect_entities(image_data_b64, request_data):
    print('YOLO Service: Detecting entities...')
    if not yolo_entity_model:
         return jsonify({"message": "YOLO entity model not loaded"}), 500

    img_cv2 = base64_to_cv2(image_data_b64)
    if img_cv2 is None:
        return jsonify({"message": "Failed to decode image data"}), 400

    try:
        results = yolo_entity_model(img_cv2, verbose=False) # Run inference
        
        detected_entities = []
        # Process results list
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].tolist() # Bounding box coordinates [x1, y1, x2, y2]
                conf = box.conf[0].item()   # Confidence score
                cls_id = int(box.cls[0].item()) # Class ID
                label = yolo_entity_model.names[cls_id] # Get class name
                
                detected_entities.append({
                    "box": xyxy,
                    "label": label,
                    "confidence": conf
                })

        print(f"YOLO Entities: Found {len(detected_entities)} objects.")
        return jsonify({"detectedEntities": detected_entities})

    except Exception as e:
        print(f"Error during YOLO entity detection: {e}")
        traceback.print_exc()
        return jsonify({"message": f"Error during detection: {e}"}), 500

# Standard COCO keypoint names corresponding to the typical YOLOv8 pose model output order
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

@app.route('/api/yolo/detect-posture', methods=['POST'])
@require_image_data
def detect_posture(image_data_b64, request_data):
    print('YOLO Service: Detecting posture...')
    if not yolo_pose_model:
         return jsonify({"message": "YOLO pose model not loaded"}), 500

    img_cv2 = base64_to_cv2(image_data_b64)
    if img_cv2 is None:
        return jsonify({"message": "Failed to decode image data"}), 400

    try:
        results = yolo_pose_model(img_cv2, verbose=False) # Run inference
        
        all_keypoints = []
        # Process results
        for result in results:
            if result.keypoints: # Check if keypoints were detected
                kpts_data = result.keypoints.data # Shape: (num_persons, num_keypoints, 3) [x, y, conf]
                num_keypoints = kpts_data.shape[1]

                # Iterate through detected persons/poses
                for person_kpts in kpts_data:
                    person_keypoints_list = []
                    for i, kpt in enumerate(person_kpts):
                        x, y, score = kpt.tolist()
                        # Use the defined COCO names based on index
                        if i < len(COCO_KEYPOINT_NAMES):
                            name = COCO_KEYPOINT_NAMES[i]
                        else:
                             name = f'kpt_{i}' # Fallback for unexpected extra keypoints

                        # Apply a confidence threshold
                        if score > 0.1:
                             person_keypoints_list.append({"x": x, "y": y, "score": score, "name": name})

                    # Only add keypoints for this person if any were above threshold
                    if person_keypoints_list:
                        # Flattening all keypoints for now, as per original logic.
                        # If per-person structure is needed, change the response format.
                        all_keypoints.extend(person_keypoints_list)

        print(f"YOLO Posture: Found {len(all_keypoints)} keypoints total (above threshold)." )
        return jsonify({"detectedPosture": {"keypoints": all_keypoints, "connections": []}})

    except Exception as e:
        print(f"Error during YOLO posture detection: {e}")
        traceback.print_exc()
        return jsonify({"message": f"Error during posture detection: {e}"}), 500

@app.route('/api/arcface/detect-faces', methods=['POST'])
@require_image_data
def detect_faces(image_data_b64, request_data):
    print('DeepFace Service: Detecting faces...')
    if not DeepFace:
         return jsonify({"message": "DeepFace library not loaded"}), 500

    img_cv2 = base64_to_cv2(image_data_b64)
    if img_cv2 is None:
        return jsonify({"message": "Failed to decode image data"}), 400

    try:
        # Use DeepFace.analyze to get bounding box and landmarks
        # Use ArcFace model for embeddings, specify detector backend if needed
        # Set actions to just 'embedding' and 'landmark' to avoid emotion/age/etc.
        # enforce_detection=False allows processing images with no faces found
        results = DeepFace.analyze(
            img_path=img_cv2,
            actions=['embedding', 'landmarks'],
            model_name='ArcFace', # Specify the model for embeddings
            detector_backend='retinaface', # Example detector, others available
            enforce_detection=False,
            silent=True # Reduce console output
        )
        
        detected_faces_output = []
        # DeepFace returns a list of dicts, one per detected face
        if isinstance(results, list):
             for face_data in results:
                # Extract region (bounding box) and landmarks
                region = face_data.get('region') # Format: {'x': int, 'y': int, 'w': int, 'h': int}
                facial_landmarks = face_data.get('landmarks') # Format: {'right_eye': [x,y], 'left_eye': ...}
                embedding = face_data.get('embedding')

                if region and facial_landmarks:
                    # Convert region to [x1, y1, x2, y2] format
                    x1, y1, w, h = region['x'], region['y'], region['w'], region['h']
                    box = [x1, y1, x1 + w, y1 + h]

                    # Convert landmarks dict to simple list [[x,y], ...]
                    # Order might vary based on detector, but we grab what's available
                    landmarks_list = list(facial_landmarks.values())

                    detected_faces_output.append({
                        "box": box,
                        "landmarks": landmarks_list,
                        "embedding": embedding
                    })

        print(f"DeepFace: Found {len(detected_faces_output)} faces.")
        return jsonify({"detectedFaces": detected_faces_output})

    except ValueError as ve:
         # Handle case where DeepFace enforce_detection=True and no face found
         if "Face could not be detected" in str(ve):
             print("DeepFace: No faces found.")
             return jsonify({"detectedFaces": []}) # Return empty list
         else:
             print(f"Error during DeepFace analysis: {ve}")
             traceback.print_exc()
             return jsonify({"message": f"Error during face analysis: {ve}"}), 500
    except Exception as e:
        print(f"Error during DeepFace analysis: {e}")
        traceback.print_exc()
        return jsonify({"message": f"Error during face analysis: {e}"}), 500

@app.route('/api/gemini/describe-image', methods=['POST'])
@require_image_data
def gemini_describe_image(image_data_b64, request_data):
    print('Gemini Service: Getting image description...')
    if not genai_model:
        return jsonify({"message": "Gemini model not initialized (check API key)"}), 500
    
    image_part = base64_to_gemini_part(image_data_b64)
    if not image_part:
        return jsonify({"message": "Could not process image data for Gemini"}), 400
        
    try:
        prompt = "Describe this image in detail."
        response = genai_model.generate_content([prompt, image_part])
        print("Gemini Description Response:", response.text)
        return jsonify({"description": response.text})
    except Exception as e:
        print(f"Error calling Gemini API for description: {e}")
        error_message = str(e)
        if hasattr(e, 'message'): error_message = e.message # Try to get more specific error
        return jsonify({"message": f"Error communicating with Gemini: {error_message}"}), 500

@app.route('/api/gemini/ask-about-image', methods=['POST'])
@require_image_data
def gemini_ask_about_image(image_data_b64, request_data):
    print('Gemini Service: Asking about image...')
    if not genai_model:
        return jsonify({"message": "Gemini model not initialized (check API key)"}), 500

    prompt = request_data.get('prompt')
    if not prompt:
         return jsonify({"message": "Missing prompt"}), 400

    image_part = base64_to_gemini_part(image_data_b64)
    if not image_part:
        return jsonify({"message": "Could not process image data for Gemini"}), 400
        
    try:
        print(f"Asking Gemini with prompt: {prompt}")
        response = genai_model.generate_content([prompt, image_part])
        print("Gemini Question Response:", response.text)
        return jsonify({"response": response.text})
    except Exception as e:
        print(f"Error calling Gemini API for question: {e}")
        error_message = str(e)
        if hasattr(e, 'message'): error_message = e.message # Try to get more specific error
        return jsonify({"message": f"Error communicating with Gemini: {error_message}"}), 500

# Placeholder function - assumes genai_model can handle image I/O prompts
# You might need to adjust model_name or generation parameters depending on API capabilities
@app.route('/api/gemini/generate-with-image', methods=['POST'])
@require_image_data
def gemini_generate_with_image(image_data_b64, request_data):
    print('Gemini Service: Generating based on image and prompt...')
    if not genai_model:
        return jsonify({"message": "Gemini model not initialized"}), 500

    prompt = request_data.get('prompt')
    if not prompt:
        return jsonify({"message": "Missing prompt"}), 400

    image_part = base64_to_gemini_part(image_data_b64)
    if not image_part:
        return jsonify({"message": "Could not process image data for Gemini"}), 400

    try:
        # Construct the prompt for image generation/modification
        # The exact format depends on what the model expects
        # Example: might involve specific instructions like "Edit this image:"
        full_prompt = [prompt, image_part]

        print(f"Generating with Gemini using prompt: {prompt}")
        # This call might need adjustment based on the specific Gemini API
        # for image generation/output. It might return structured data
        # containing image bytes or a URL, not just text.
        response = genai_model.generate_content(full_prompt)

        # --- Process the response ---
        # This part is HIGHLY dependent on the actual Gemini API response format
        # for image generation. We'll assume it might have 'text' or image data.

        response_text = None
        response_image_b64 = None

        # Attempt to extract text (standard response)
        try:
            response_text = response.text
            print("Gemini Generation Response (Text):", response_text)
        except Exception:
             # Ignore if text extraction fails (might be image-only response)
             pass

        # TODO: Attempt to extract image data if the API supports it.
        # This is a placeholder - you'll need to adapt based on Gemini's actual response.
        # Example: If response contains image bytes in a specific part:
        # if hasattr(response, 'parts') and len(response.parts) > 0:
        #     img_part = response.parts[0] # Assuming the first part is the image
        #     if hasattr(img_part, 'blob') and img_part.blob.mime_type.startswith('image/'):
        #         img_bytes = img_part.blob.data
        #         response_image_b64 = f"data:{img_part.blob.mime_type};base64," + base64.b64encode(img_bytes).decode('utf-8')
        #         response_text = None # Clear text if we got an image
        #         print("Gemini Generation Response (Image Received)")

        # If no image was explicitly generated, return the text response
        if response_image_b64:
             return jsonify({"response_text": None, "response_image_b64": response_image_b64})
        elif response_text:
             return jsonify({"response_text": response_text, "response_image_b64": None})
        else:
             # Handle cases where response is empty or format is unexpected
             print("Warning: Gemini generation response was empty or in unexpected format.")
             return jsonify({"response_text": "Gemini returned an empty or unexpected response.", "response_image_b64": None})

    except Exception as e:
        print(f"Error calling Gemini API for generation: {e}")
        error_message = str(e)
        # Attempt to get more specific error details if available
        if hasattr(e, 'message'): error_message = e.message
        elif hasattr(e, 'response') and hasattr(e.response, 'text'): error_message = e.response.text
        return jsonify({"message": f"Error communicating with Gemini: {error_message}"}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on the network if needed
    # Use debug=True for development (auto-reloads), but disable for production
    # Default Flask port is 5000, but frontend expects 3001
    app.run(host='0.0.0.0', port=3001, debug=True) 