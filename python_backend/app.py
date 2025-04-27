import os
import base64
import io
import random
import time
from functools import wraps
import traceback

import google.genai as genai
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from PIL import Image
import cv2 # OpenCV for image processing
from google.genai import types as genai_types # Import types at module level

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
MODEL_NAME = "gemini-1.5-flash-latest"  # Use a stable, recent model
GEMINI_IMAGE_GEN_MODEL_NAME = "gemini-2.0-flash-exp-image-generation" # Use the same model if it supports image generation, or a dedicated one like "imagen" if needed and available
# Note: The specific 'gemini-2.0-flash-exp-image-generation' might be deprecated or replaced.
# Using a standard multimodal model like 1.5 Flash is often sufficient for both text/vision and basic image generation/editing.
# If advanced image generation (like Imagen) is needed, a different client/method might be required.

client = None
if GEMINI_API_KEY:
    try:
        # Initialize the client - it reads GOOGLE_API_KEY env var automatically
        # Or pass explicitly: client = genai.Client(api_key=GEMINI_API_KEY)
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini Client Initialized Successfully.")
    except Exception as e:
        print(f"Error initializing Gemini Client: {e}")
        client = None # Ensure client is None if initialization fails
else:
    print("Gemini client not initialized due to missing API key.")

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
    print('DeepFace Service: Detecting faces and extracting embeddings...')
    if not DeepFace:
         return jsonify({"message": "DeepFace library not loaded"}), 500

    img_cv2 = base64_to_cv2(image_data_b64)
    if img_cv2 is None:
        return jsonify({"message": "Failed to decode image data"}), 400

    try:
        # 1. Extract faces using a detector that provides landmarks (like retinaface)
        # Setting align=True is recommended for better embedding quality with represent()
        extracted_faces_info = DeepFace.extract_faces(
            img_path=img_cv2,
            detector_backend='retinaface',
            enforce_detection=False, # Don't error if no face found
            align=True # Get aligned face chips for represent()
        )

        detected_faces_output = []

        # Check if extract_faces returned any results
        if not isinstance(extracted_faces_info, list):
            print("Warning: DeepFace.extract_faces did not return a list as expected.")
            extracted_faces_info = [] # Treat as empty list to avoid errors

        # 2. Iterate through extracted faces and get embeddings for each
        for face_info in extracted_faces_info:
            # Ensure we have the necessary info from extract_faces
            face_img_np = face_info.get('face') # Aligned face chip as numpy array
            facial_area = face_info.get('facial_area') # Original bounding box {'x', 'y', 'w', 'h'}
            landmarks = face_info.get('landmarks') # Landmarks {'right_eye': [x,y], ...}
            confidence = face_info.get('confidence') # Detection confidence

            # Skip if essential info is missing (e.g., align=False might not return 'face')
            if face_img_np is None or facial_area is None:
                 print(f"Warning: Skipping face due to missing 'face' or 'facial_area'. Confidence: {confidence}")
                 continue

            try:
                 # --- DEBUG: Save the aligned face chip being processed ---
                 # Create a unique filename
                 timestamp = time.time()
                 debug_face_filename = f"debug_face_{timestamp}_{facial_area['x']}_{facial_area['y']}.jpg"
                 # Ensure the input is in the correct format for imwrite (BGR, uint8)
                 # DeepFace usually returns BGR numpy arrays
                 if face_img_np.ndim == 3 and face_img_np.shape[2] == 3:
                     try:
                        # Convert float image (often 0-1) back to uint8 (0-255) if needed
                        if face_img_np.dtype == np.float32 or face_img_np.dtype == np.float64:
                             face_to_save = (face_img_np * 255).astype(np.uint8)
                        else:
                             face_to_save = face_img_np.astype(np.uint8) # Ensure uint8

                        save_success = cv2.imwrite(debug_face_filename, face_to_save)
                        if save_success:
                             print(f"DEBUG: Saved aligned face chip to {debug_face_filename}")
                        else:
                             print(f"DEBUG: FAILED to save aligned face chip to {debug_face_filename}")
                     except Exception as save_err:
                        print(f"DEBUG: Error saving face chip {debug_face_filename}: {save_err}")
                 else:
                      print(f"DEBUG: Cannot save face chip, unexpected shape/dims: {face_img_np.shape}, dtype: {face_img_np.dtype}")
                 # --- END DEBUG ---

                 # 3. Get embedding for the ALIGNED face chip
                 embedding_objs = DeepFace.represent(
                     img_path=face_img_np,
                     model_name='VGG-Face', # Changed model again
                     enforce_detection=False # We already know it's a face
                 )
                 
                 # represent() usually returns a list containing one dict for the single face chip
                 if not embedding_objs or not isinstance(embedding_objs, list) or len(embedding_objs) == 0 or 'embedding' not in embedding_objs[0]:
                      print(f"Warning: Could not get embedding for face at {facial_area}. Skipping.")
                      continue
                 embedding = embedding_objs[0]['embedding']

                 # --- DEBUG LOGGING: Print first few embedding values ---
                 print(f"DEBUG: Embedding generated for face at {facial_area}: {str(embedding[:5])}...") 
                 # --- END DEBUG LOGGING ---

                 # 4. Format the output for the frontend
                 # Convert bounding box from {'x', 'y', 'w', 'h'} to [x1, y1, x2, y2]
                 x1, y1, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                 box = [x1, y1, x1 + w, y1 + h]

                 # Convert landmarks dict to list [[x,y], ...]
                 landmarks_list = list(landmarks.values()) if landmarks and isinstance(landmarks, dict) else []

                 detected_faces_output.append({
                    "box": box,
                    "landmarks": landmarks_list,
                    "embedding": embedding,
                    "confidence": confidence # Include detection confidence
                 })

            except Exception as represent_err:
                 # Log error during represent() but continue if possible
                 print(f"Error during DeepFace.represent for face at {facial_area}: {represent_err}")
                 traceback.print_exc()

        print(f"DeepFace: Successfully processed {len(detected_faces_output)} faces with embeddings.")
        return jsonify({"detectedFaces": detected_faces_output})

    except Exception as e:
        # Catch errors during extract_faces or other unexpected issues
        print(f"Error during DeepFace face detection/embedding: {e}")
        traceback.print_exc()
        # Check if it's the specific "no face detected" error when enforce_detection=True (though we set it False)
        if "Face could not be detected" in str(e):
             print("DeepFace (extract_faces): No faces found.")
             return jsonify({"detectedFaces": []})
        return jsonify({"message": f"Error during face processing: {e}"}), 500

@app.route('/api/gemini/describe-image', methods=['POST'])
@require_image_data
def gemini_describe_image(image_data_b64, request_data):
    print('Gemini Service: Getting image description...')
    if not client:
        return jsonify({"message": "Gemini client not initialized (check API key)"}), 500

    image_part_dict = base64_to_gemini_part(image_data_b64)
    if not image_part_dict:
        return jsonify({"message": "Could not process image data for Gemini"}), 400

    try:
        # Create image content using Part/inline_data - multiple options to try
        # Option 1: Direct construction with keyword arguments
        image_part_obj = genai_types.Part(
            inline_data=genai_types.Blob(
                mime_type=image_part_dict['mime_type'],
                data=base64.b64decode(image_part_dict['data'])
            )
        )

        prompt = "Describe this image in detail."
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image_part_obj] # Pass the Part object
        )
        print("Gemini Description Response:", response.text)
        return jsonify({"description": response.text})
    except base64.binascii.Error as b64_error:
        print(f"Error decoding base64 data: {b64_error}")
        return jsonify({"message": f"Invalid base64 image data: {b64_error}"}), 400
    except Exception as e:
        print(f"Error calling Gemini API for description: {e}")
        error_message = str(e)
        # Attempt to parse Pydantic validation errors for better feedback
        if "validation error" in error_message.lower():
             try:
                 import json
                 details = json.loads(error_message.split('\n', 1)[1] if '\n' in error_message else error_message)
                 first_error = details[0]['msg'] if isinstance(details, list) and details else "Invalid input structure"
                 error_message = f"Input Validation Error: {first_error}"
             except: # Fallback if parsing fails
                 pass # Keep original error message
        elif hasattr(e, 'message'): error_message = e.message
        traceback.print_exc()
        return jsonify({"message": f"Error communicating with Gemini: {error_message}"}), 500

@app.route('/api/gemini/ask-about-image', methods=['POST'])
@require_image_data
def gemini_ask_about_image(image_data_b64, request_data):
    print('Gemini Service: Asking about image...')
    if not client:
        return jsonify({"message": "Gemini client not initialized (check API key)"}), 500

    prompt = request_data.get('prompt')
    if not prompt:
         return jsonify({"message": "Missing prompt"}), 400

    image_part_dict = base64_to_gemini_part(image_data_b64)
    if not image_part_dict:
        return jsonify({"message": "Could not process image data for Gemini"}), 400

    try:
        # Create image content using Part/inline_data
        image_part_obj = genai_types.Part(
            inline_data=genai_types.Blob(
                mime_type=image_part_dict['mime_type'],
                data=base64.b64decode(image_part_dict['data'])
            )
        )

        print(f"Asking Gemini with prompt: {prompt}")
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image_part_obj] # Pass the Part object
        )
        print("Gemini Question Response:", response.text)
        return jsonify({"response": response.text})
    except base64.binascii.Error as b64_error:
        print(f"Error decoding base64 data: {b64_error}")
        return jsonify({"message": f"Invalid base64 image data: {b64_error}"}), 400
    except Exception as e:
        print(f"Error calling Gemini API for question: {e}")
        error_message = str(e)
        if "validation error" in error_message.lower():
             try:
                 import json
                 details = json.loads(error_message.split('\n', 1)[1] if '\n' in error_message else error_message)
                 first_error = details[0]['msg'] if isinstance(details, list) and details else "Invalid input structure"
                 error_message = f"Input Validation Error: {first_error}"
             except: pass
        elif hasattr(e, 'message'): error_message = e.message
        traceback.print_exc()
        return jsonify({"message": f"Error communicating with Gemini: {error_message}"}), 500

@app.route('/api/gemini/generate-with-image', methods=['POST'])
@require_image_data
def gemini_generate_with_image(image_data_b64, request_data):
    print('Gemini Service: Generating based on image and prompt...')
    if not client:
        return jsonify({"message": "Gemini client not initialized"}), 500

    prompt = request_data.get('prompt')
    if not prompt:
        return jsonify({"message": "Missing prompt"}), 400

    image_part_dict = base64_to_gemini_part(image_data_b64)
    if not image_part_dict:
        return jsonify({"message": "Could not process image data for Gemini"}), 400

    try:
        # Create image content using Part/inline_data
        image_part_obj = genai_types.Part(
            inline_data=genai_types.Blob(
                mime_type=image_part_dict['mime_type'],
                data=base64.b64decode(image_part_dict['data'])
            )
        )

        full_prompt = [prompt, image_part_obj] # Pass text prompt and Part object
        print(f"Generating with Gemini using prompt: {prompt}")

        # Try without the GenerationConfig first
        response = client.models.generate_content(
            model=GEMINI_IMAGE_GEN_MODEL_NAME,
            contents=full_prompt,
            config=genai_types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )

        response_text = None
        response_image_b64 = None
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                response_text = part.text
            elif part.inline_data is not None:
                b64_encoded = base64.b64encode(part.inline_data.data).decode('utf-8')
                response_image_b64 = f"data:{part.inline_data.mime_type};base64,{b64_encoded}"
                response_text = None # Prioritize image if found

        if response_image_b64:
            print("Returning generated image.")
            return jsonify({"response_text": None, "response_image_b64": response_image_b64})
        elif response_text:
            print("Returning generated text.")
            return jsonify({"response_text": response_text, "response_image_b64": None})
        else:
            print("Warning: Gemini generation response was empty or contained no recognizable parts.")
            # Check for safety feedback or other reasons for empty response
            safety_feedback = getattr(response, 'prompt_feedback', None)
            if safety_feedback and safety_feedback.block_reason:
                 error_msg = f"Blocked by safety filter: {safety_feedback.block_reason.name}"
                 print(error_msg)
                 return jsonify({"message": error_msg}), 400 # Return appropriate status
            else:
                 print(response)
                 return jsonify({"response_text": "Gemini returned an empty or unexpected response.", "response_image_b64": None})

    except Exception as e:
        print(f"Error calling Gemini API for generation: {e}")
        error_message = str(e)
        if hasattr(e, 'message'): error_message = e.message
        elif hasattr(e, 'response') and hasattr(e.response, 'text'): error_message = e.response.text
        traceback.print_exc()
        return jsonify({"message": f"Error communicating with Gemini: {error_message}"}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on the network if needed
    # Use debug=True for development (auto-reloads), but disable for production
    # Default Flask port is 5000, but frontend expects 3001
    app.run(host='0.0.0.0', port=3001, debug=True) 