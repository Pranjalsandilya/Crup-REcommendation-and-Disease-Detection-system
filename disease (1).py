from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import json
import os
from backend.schemas.disease_schema import DiseaseResponse
from backend.services.gemini_service import get_disease_insights
from backend.utils.image_processing import preprocess_image

try:
    import tensorflow as tf
except ImportError:
    tf = None

router = APIRouter()

MODEL_PATH = "backend/models/plant_disease_prediction_model.h5"
CLASS_INDICES_PATH = "backend/models/class_indices.json"
model = None
class_indices = {}

def load_resources():
    global model, class_indices
    # Load Model
    if tf and os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Disease model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading disease model: {e}")
    else:
        if not tf:
            print("TensorFlow not installed. Using Gemini Vision API for disease detection.")
        else:
            print(f"Disease model not found at {MODEL_PATH}")

    # Load Class Indices
    if os.path.exists(CLASS_INDICES_PATH):
        try:
            with open(CLASS_INDICES_PATH, 'r') as f:
                class_indices = json.load(f)
            print(f"Class indices loaded from {CLASS_INDICES_PATH}")
        except Exception as e:
            print(f"Error loading class indices: {e}")
    else:
        print(f"Class indices not found at {CLASS_INDICES_PATH}")

load_resources()

@router.post("/detect", response_model=DiseaseResponse)
async def detect_disease(file: UploadFile = File(...), language: str = "en-IN"):
    """Detect plant disease using TensorFlow model"""
    
    # Check if TensorFlow and model are available
    if not tf:
        return DiseaseResponse(
            disease="TensorFlow Not Available",
            details={
                "description": "Disease detection requires TensorFlow, which is not installed. This is due to Python version compatibility (TensorFlow supports Python 3.9-3.11, you're using Python 3.14).",
                "treatment": "To enable ML-based disease detection: Install Python 3.9-3.11, then run 'pip install tensorflow'. For now, you can use the general guidance provided below.",
                "prevention": "General disease prevention: Use disease-resistant varieties, maintain proper plant spacing (30-45cm), ensure good air circulation, practice crop rotation, remove infected plant debris, avoid overhead watering, and monitor plants weekly for early symptoms.",
                "quick_actions": "• Take clear photos of affected leaves\n• Compare symptoms with common diseases online\n• Consult local agricultural extension office\n• Isolate affected plants to prevent spread"
            }
        )
    
    if not model or not class_indices:
        return DiseaseResponse(
            disease="Model Not Loaded",
            details={
                "description": "The disease detection model files are missing. Please ensure plant_disease_prediction_model.h5 and class_indices.json exist in backend/models/ directory.",
                "treatment": "General treatment: Remove infected plant parts, apply appropriate fungicide (Mancozeb 2.5g/liter or Copper Oxychloride 3g/liter) every 7-10 days, ensure good air circulation.",
                "prevention": "Use disease-free seeds, practice crop rotation, maintain proper spacing, monitor regularly, and apply preventive fungicide during disease-prone seasons.",
                "quick_actions": "• Remove infected parts immediately\n• Improve air circulation\n• Consult agricultural expert\n• Monitor nearby plants"
            }
        )

    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        
        # Get disease name
        disease_name = class_indices.get(str(predicted_index), "Unknown Disease")
        disease_name_clean = disease_name.replace("___", " - ").replace("_", " ")
        
        # Get AI insights for this disease
        insights = await get_disease_insights(disease_name_clean, language=language)
        
        # Add confidence score to description
        if "description" in insights:
            insights["description"] = f"🤖 ML Detection (Confidence: {confidence*100:.1f}%). {insights['description']}"
        
        return DiseaseResponse(
            disease=disease_name_clean,
            details=insights
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

