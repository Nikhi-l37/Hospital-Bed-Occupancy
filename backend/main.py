from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
import logging
import sys
import re
import easyocr
import cv2
import numpy as np
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hospital Bed Occupancy API")

# Allow the React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once when the server starts
MODEL_PATH = "hospital_bed_model.pkl"
TOTAL_BEDS = 150

model = None
try:
    model = joblib.load(MODEL_PATH)
    logger.info("✅ Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")

# Initialize EasyOCR reader once at startup (downloads model on first run)
logger.info("⏳ Initializing EasyOCR reader...")
try:
    ocr_reader = easyocr.Reader(['en'])
    logger.info("✅ EasyOCR reader initialized successfully!")
except Exception as e:
    ocr_reader = None
    logger.error(f"❌ Error initializing EasyOCR: {e}")

@app.get("/")
def home():
    logger.info("Health check endpoint called")
    return {"message": "Hospital API is running!"}

@app.get("/predict")
def predict_occupancy(date: str = Query(..., description="Date in YYYY-MM-DD format")):
    if model is None:
        logger.error("Predict called but model is not loaded")
        return {"error": "Model not loaded. Please checked backend logs."}

    try:
        logger.info(f"Prediction requested for 7 days starting: {date}")
        
        # Validate date format
        try:
            start_date = pd.to_datetime(date)
        except ValueError:
            logger.warning(f"Invalid date format received: {date}")
            return {"error": "Invalid date format. Use YYYY-MM-DD."}

        predictions = []

        # Generate predictions for 7 days
        for i in range(7):
            current_date = start_date + pd.Timedelta(days=i)
            
            # Prepare the input for the model
            future = pd.DataFrame({
                'ds': [current_date],
                'is_holiday': [1 if current_date.dayofweek == 6 else 0] # Match your model logic
            })
            
            forecast = model.predict(future)
            prediction = forecast.iloc[0]
            
            occupancy = int(prediction['yhat'])
            upper_bound = int(prediction['yhat_upper']) # Worst-case scenario
            
            available = TOTAL_BEDS - occupancy
            risk_level = "CRITICAL" if (TOTAL_BEDS - upper_bound) < 15 else "LOW"

            predictions.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "predicted_occupancy": occupancy,
                "worst_case": upper_bound,
                "available_beds": available,
                "risk": risk_level
            })
        
        logger.info(f"Generated {len(predictions)} predictions successfully")
        return predictions

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return {"error": str(e)}


# ─── Module 1: Vision System (OCR Prescription Scanner) ─────────────────

@app.post("/upload-prescription")
async def upload_prescription(file: UploadFile = File(...)):
    """Accept a prescription image and extract text using EasyOCR."""
    if ocr_reader is None:
        raise HTTPException(status_code=500, detail="OCR reader is not initialized.")

    try:
        # Read uploaded image bytes
        contents = await file.read()
        logger.info(f"Received image: {file.filename} ({len(contents)} bytes)")

        # Decode image bytes to a NumPy array for OpenCV
        np_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file. Could not decode.")

        # Run EasyOCR on the image
        results = ocr_reader.readtext(image, detail=0)  # detail=0 returns text only
        logger.info(f"Extracted {len(results)} text segments from {file.filename}")

        return {"extracted_text": results}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


# ─── Modules 2 & 3: NLP Symptom Analyzer & Recommendation Engine ────────

# Pydantic model for the request body
class SymptomRequest(BaseModel):
    patient_text: str

# Mock medicine database: keyword -> treatment
MEDICINE_DB = {
    "fever": {
        "medicine": "Paracetamol 500mg",
        "timeline": "1 pill every 8 hours for 3 days"
    },
    "headache": {
        "medicine": "Aspirin 325mg",
        "timeline": "1 pill every 6 hours as needed"
    },
    "infection": {
        "medicine": "Amoxicillin 500mg",
        "timeline": "1 pill every 12 hours for 5 days"
    },
    "cough": {
        "medicine": "Dextromethorphan 15mg",
        "timeline": "1 dose every 6-8 hours"
    },
    "cold": {
        "medicine": "Cetirizine 10mg",
        "timeline": "1 pill daily for 5 days"
    },
    "sore throat": {
        "medicine": "Ibuprofen 400mg",
        "timeline": "1 pill every 8 hours with food"
    },
    "nausea": {
        "medicine": "Ondansetron 4mg",
        "timeline": "1 pill every 8 hours as needed"
    },
    "diarrhea": {
        "medicine": "ORS Sachets + Loperamide 2mg",
        "timeline": "ORS after every loose stool; Loperamide 1 pill then 1 after each episode (max 4/day)"
    },
    "allergy": {
        "medicine": "Loratadine 10mg",
        "timeline": "1 pill daily"
    },
    "body pain": {
        "medicine": "Diclofenac 50mg",
        "timeline": "1 pill every 8 hours after meals"
    },
}


@app.post("/analyze-symptoms")
def analyze_symptoms(data: SymptomRequest):
    """Scan patient text for symptom keywords and return matched treatments."""
    text = data.patient_text.lower()
    logger.info(f"Analyzing symptoms from text: '{data.patient_text[:80]}...'")

    detected_symptoms = []
    recommendations = []

    for keyword, treatment in MEDICINE_DB.items():
        # Use regex word-boundary matching for accurate detection
        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
            detected_symptoms.append(keyword)
            recommendations.append({
                "symptom": keyword,
                "medicine": treatment["medicine"],
                "timeline": treatment["timeline"]
            })

    logger.info(f"Detected {len(detected_symptoms)} symptoms: {detected_symptoms}")

    return {
        "detected_symptoms": detected_symptoms,
        "recommendations": recommendations
    }

if __name__ == "__main__":
    import uvicorn
    # Verify port availability before running could be added here, 
    # but uvicorn handles it by crashing if port is taken.
    uvicorn.run(app, host="0.0.0.0", port=8000)