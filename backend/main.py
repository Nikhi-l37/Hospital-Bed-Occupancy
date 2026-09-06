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
import json

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

# ─── Expanded Medicine Database with Urgency Levels ─────────────────────
# urgency: "mild" (🟢 self-treat), "moderate" (🟡 see doctor in 24h), "severe" (🔴 immediate)
MEDICINE_DB = {
    "fever": {
        "medicine": "Paracetamol 500mg",
        "timeline": "1 pill every 8 hours for 3 days",
        "urgency": "moderate",
        "advice": "Stay hydrated and rest. If fever persists beyond 3 days, consult a doctor."
    },
    "headache": {
        "medicine": "Aspirin 325mg or Paracetamol 500mg",
        "timeline": "1 pill every 6 hours as needed",
        "urgency": "mild",
        "advice": "Rest in a dark room. Drink water. Avoid screen time."
    },
    "infection": {
        "medicine": "Amoxicillin 500mg (prescription required)",
        "timeline": "1 pill every 12 hours for 5 days",
        "urgency": "moderate",
        "advice": "Antibiotics need a doctor's prescription. Please visit a clinic."
    },
    "cough": {
        "medicine": "Dextromethorphan 15mg syrup",
        "timeline": "1 dose every 6-8 hours",
        "urgency": "mild",
        "advice": "Drink warm water with honey. Avoid cold drinks."
    },
    "cold": {
        "medicine": "Cetirizine 10mg",
        "timeline": "1 pill daily for 5 days",
        "urgency": "mild",
        "advice": "Stay warm, drink hot fluids, and take steam inhalation."
    },
    "sore throat": {
        "medicine": "Ibuprofen 400mg + warm salt water gargle",
        "timeline": "1 pill every 8 hours with food; gargle 3 times daily",
        "urgency": "mild",
        "advice": "Avoid spicy food. Drink warm liquids."
    },
    "nausea": {
        "medicine": "Ondansetron 4mg",
        "timeline": "1 pill every 8 hours as needed",
        "urgency": "mild",
        "advice": "Eat small, bland meals. Avoid oily food."
    },
    "diarrhea": {
        "medicine": "ORS Sachets + Loperamide 2mg",
        "timeline": "ORS after every loose stool; Loperamide 1 pill then 1 after each episode (max 4/day)",
        "urgency": "moderate",
        "advice": "Stay hydrated. Avoid dairy and spicy food. See a doctor if blood in stool."
    },
    "allergy": {
        "medicine": "Loratadine 10mg (Claritin)",
        "timeline": "1 pill daily",
        "urgency": "mild",
        "advice": "Identify and avoid allergens. Use antihistamines as needed."
    },
    "body pain": {
        "medicine": "Diclofenac 50mg or Ibuprofen 400mg",
        "timeline": "1 pill every 8 hours after meals",
        "urgency": "mild",
        "advice": "Rest, apply warm compress, and avoid heavy lifting."
    },
    "vomiting": {
        "medicine": "Ondansetron 4mg + ORS",
        "timeline": "1 pill every 8 hours; ORS sips frequently",
        "urgency": "moderate",
        "advice": "Don't eat solid food for a few hours. Sip ORS slowly. See a doctor if persistent."
    },
    "stomach pain": {
        "medicine": "Pantoprazole 40mg + Dicyclomine 20mg",
        "timeline": "Pantoprazole: 1 pill before breakfast; Dicyclomine: as needed for cramps",
        "urgency": "moderate",
        "advice": "Avoid spicy, oily, and acidic food. Eat small meals."
    },
    "back pain": {
        "medicine": "Diclofenac 50mg + muscle relaxant (Thiocolchicoside)",
        "timeline": "1 pill every 8 hours after meals for 3-5 days",
        "urgency": "mild",
        "advice": "Use a hot water bag. Avoid bending. Maintain good posture."
    },
    "dizziness": {
        "medicine": "Meclizine 25mg",
        "timeline": "1 pill every 8 hours as needed",
        "urgency": "moderate",
        "advice": "Sit or lie down when dizzy. Avoid sudden movements. Stay hydrated."
    },
    "rapid heartbeat": {
        "medicine": "⚠️ Do NOT self-medicate",
        "timeline": "Seek immediate medical attention",
        "urgency": "severe",
        "advice": "🚨 Contact a doctor or visit the emergency room IMMEDIATELY. Rapid heartbeat can indicate a serious cardiac condition."
    },
    "chest pain": {
        "medicine": "⚠️ EMERGENCY — Call 108 / 112 immediately",
        "timeline": "Do not wait — seek emergency care NOW",
        "urgency": "severe",
        "advice": "🚨 Chest pain can be a sign of a heart attack. Call emergency services (108) immediately. Chew an Aspirin 325mg ONLY if not allergic."
    },
    "breathlessness": {
        "medicine": "⚠️ Do NOT self-medicate",
        "timeline": "Seek immediate medical attention",
        "urgency": "severe",
        "advice": "🚨 Difficulty breathing requires urgent medical care. Sit upright, stay calm, and call for help immediately."
    },
    "anxiety": {
        "medicine": "Deep breathing + consult a doctor for medication",
        "timeline": "Practice 4-7-8 breathing technique daily",
        "urgency": "moderate",
        "advice": "Try relaxation exercises. Reduce caffeine intake. Consider speaking to a mental health professional."
    },
    "insomnia": {
        "medicine": "Melatonin 3mg (OTC)",
        "timeline": "1 tablet 30 minutes before bedtime",
        "urgency": "mild",
        "advice": "Avoid screens 1 hour before bed. Keep a regular sleep schedule."
    },
    "joint pain": {
        "medicine": "Diclofenac gel (topical) + Glucosamine supplement",
        "timeline": "Apply gel 3 times daily; 1 Glucosamine tablet daily",
        "urgency": "mild",
        "advice": "Gentle stretching helps. Avoid overexertion. See a doctor if swelling persists."
    },
    "skin rash": {
        "medicine": "Calamine lotion + Cetirizine 10mg",
        "timeline": "Apply lotion as needed; 1 Cetirizine pill daily",
        "urgency": "mild",
        "advice": "Keep the area clean and dry. Avoid scratching. See a dermatologist if spreading."
    },
    "acidity": {
        "medicine": "Pantoprazole 40mg or Ranitidine 150mg",
        "timeline": "1 pill before breakfast daily",
        "urgency": "mild",
        "advice": "Avoid spicy food, caffeine, and late-night meals. Eat smaller portions."
    },
    "constipation": {
        "medicine": "Isabgol (Psyllium husk) + Lactulose syrup",
        "timeline": "Isabgol: 2 tsp in water at bedtime; Lactulose: 15ml at night",
        "urgency": "mild",
        "advice": "Drink more water. Eat fiber-rich fruits and vegetables. Exercise regularly."
    },
    "muscle cramps": {
        "medicine": "Magnesium supplement + Diclofenac gel",
        "timeline": "1 Magnesium tablet daily; apply gel as needed",
        "urgency": "mild",
        "advice": "Stay hydrated. Stretch before exercise. Eat bananas for potassium."
    },
    "eye strain": {
        "medicine": "Artificial tears (eye drops)",
        "timeline": "1-2 drops every 4 hours as needed",
        "urgency": "mild",
        "advice": "Follow the 20-20-20 rule: every 20 minutes, look 20 feet away for 20 seconds."
    },
    "toothache": {
        "medicine": "Ibuprofen 400mg + Clove oil (topical)",
        "timeline": "1 Ibuprofen every 8 hours; apply clove oil on affected area",
        "urgency": "moderate",
        "advice": "Visit a dentist as soon as possible. Rinse with warm salt water."
    },
    "ear pain": {
        "medicine": "Ibuprofen 400mg (for pain) + see ENT doctor",
        "timeline": "1 pill every 8 hours; see doctor within 24 hours",
        "urgency": "moderate",
        "advice": "Do not insert anything into the ear. Keep it dry. See an ENT specialist."
    },
    "high blood pressure": {
        "medicine": "⚠️ Requires doctor-prescribed medication",
        "timeline": "Consult a doctor for proper diagnosis and treatment plan",
        "urgency": "severe",
        "advice": "🚨 High blood pressure needs medical management. Reduce salt intake, exercise, and see a doctor for medication."
    },
    "fainting": {
        "medicine": "⚠️ Seek medical attention",
        "timeline": "Visit a doctor immediately after fainting episode",
        "urgency": "severe",
        "advice": "🚨 Lie down with legs elevated. If someone faints, check breathing and call for help."
    },
}

# ─── Synonym Map: Natural language → DB keywords ───────────────────────
SYNONYM_MAP = {
    # Diarrhea synonyms
    "loose motions": "diarrhea",
    "loose motion": "diarrhea",
    "loose stools": "diarrhea",
    "watery stool": "diarrhea",
    "frequent stools": "diarrhea",
    "runny stomach": "diarrhea",
    # Heartbeat / cardiac
    "heart beating fast": "rapid heartbeat",
    "heart is racing": "rapid heartbeat",
    "heart rate high": "rapid heartbeat",
    "heart pounding": "rapid heartbeat",
    "palpitations": "rapid heartbeat",
    "fast heartbeat": "rapid heartbeat",
    "fast pulse": "rapid heartbeat",
    # Body pain
    "body pains": "body pain",
    "body ache": "body pain",
    "body aches": "body pain",
    "full body pain": "body pain",
    # Fever
    "high temperature": "fever",
    "feeling hot": "fever",
    "burning up": "fever",
    "high fever": "fever",
    # Cold
    "running nose": "cold",
    "runny nose": "cold",
    "stuffy nose": "cold",
    "nasal congestion": "cold",
    "blocked nose": "cold",
    "sneezing": "cold",
    # Nausea / Vomiting
    "throwing up": "vomiting",
    "feeling like vomiting": "nausea",
    "feel like throwing up": "nausea",
    "want to vomit": "nausea",
    "feeling sick": "nausea",
    "queasy": "nausea",
    # Stomach
    "tummy pain": "stomach pain",
    "stomach ache": "stomach pain",
    "stomach cramps": "stomach pain",
    "abdomen pain": "stomach pain",
    "abdominal pain": "stomach pain",
    "belly pain": "stomach pain",
    # Breathing
    "difficulty breathing": "breathlessness",
    "hard to breathe": "breathlessness",
    "can't breathe": "breathlessness",
    "shortness of breath": "breathlessness",
    "short of breath": "breathlessness",
    "gasping": "breathlessness",
    # Chest
    "chest tightness": "chest pain",
    "chest discomfort": "chest pain",
    "pain in chest": "chest pain",
    # Headache
    "head pain": "headache",
    "head hurts": "headache",
    "migraine": "headache",
    "head is paining": "headache",
    # Sleep
    "can't sleep": "insomnia",
    "unable to sleep": "insomnia",
    "not sleeping": "insomnia",
    "sleepless": "insomnia",
    "trouble sleeping": "insomnia",
    # Dizziness
    "feeling dizzy": "dizziness",
    "head spinning": "dizziness",
    "light headed": "dizziness",
    "lightheaded": "dizziness",
    "vertigo": "dizziness",
    # Anxiety
    "feeling anxious": "anxiety",
    "panic attack": "anxiety",
    "feeling nervous": "anxiety",
    "stressed out": "anxiety",
    "feeling restless": "anxiety",
    # Cough
    "dry cough": "cough",
    "wet cough": "cough",
    "continuous cough": "cough",
    "coughing": "cough",
    # Back pain
    "lower back pain": "back pain",
    "upper back pain": "back pain",
    "spine pain": "back pain",
    # Joints
    "knee pain": "joint pain",
    "shoulder pain": "joint pain",
    "elbow pain": "joint pain",
    "wrist pain": "joint pain",
    # Acidity
    "acid reflux": "acidity",
    "heartburn": "acidity",
    "gas problem": "acidity",
    "bloating": "acidity",
    # Skin
    "itching": "skin rash",
    "redness on skin": "skin rash",
    "skin irritation": "skin rash",
    "hives": "skin rash",
    # Constipation
    "unable to pass stool": "constipation",
    "hard stool": "constipation",
    "not passing motion": "constipation",
    # Eye
    "eyes hurting": "eye strain",
    "blurry vision": "eye strain",
    "eye pain": "eye strain",
    # Tooth
    "tooth pain": "toothache",
    "teeth hurting": "toothache",
    "dental pain": "toothache",
    # Ear
    "ear hurts": "ear pain",
    "earache": "ear pain",
    # BP
    "bp high": "high blood pressure",
    "hypertension": "high blood pressure",
    "blood pressure high": "high blood pressure",
    # Fainting
    "passed out": "fainting",
    "blacked out": "fainting",
    "lost consciousness": "fainting",
    # Muscle
    "leg cramps": "muscle cramps",
    "muscle spasm": "muscle cramps",
    "cramps": "muscle cramps",
    # Allergy
    "allergic reaction": "allergy",
    "allergic": "allergy",
    # Sore throat
    "throat pain": "sore throat",
    "throat hurts": "sore throat",
    "painful throat": "sore throat",
    "scratchy throat": "sore throat",
}


@app.post("/analyze-symptoms")
def analyze_symptoms(data: SymptomRequest):
    """Scan patient text for symptom keywords and synonyms, return matched treatments with urgency."""
    text = data.patient_text.lower()
    logger.info(f"Analyzing symptoms from text: '{data.patient_text[:80]}...'")

    detected_symptoms = set()
    recommendations = []

    # Step 1: Expand text using synonym map (replace natural phrases with DB keywords)
    expanded_text = text
    for phrase, keyword in SYNONYM_MAP.items():
        if phrase in expanded_text:
            detected_symptoms.add(keyword)

    # Step 2: Direct keyword matching on original + expanded text
    for keyword in MEDICINE_DB:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
            detected_symptoms.add(keyword)

    # Step 3: Build recommendations from detected symptoms
    has_severe = False
    for symptom in sorted(detected_symptoms):
        treatment = MEDICINE_DB[symptom]
        urgency = treatment.get("urgency", "mild")
        if urgency == "severe":
            has_severe = True
        recommendations.append({
            "symptom": symptom,
            "medicine": treatment["medicine"],
            "timeline": treatment["timeline"],
            "urgency": urgency,
            "advice": treatment.get("advice", "")
        })

    logger.info(f"Detected {len(detected_symptoms)} symptoms: {list(detected_symptoms)}")

    # General advice when nothing is detected
    general_advice = ""
    if len(detected_symptoms) == 0:
        general_advice = "We couldn't identify specific symptoms from your description. Please try describing your condition differently, or consult a healthcare professional for proper diagnosis."

    return {
        "detected_symptoms": list(detected_symptoms),
        "recommendations": recommendations,
        "has_severe": has_severe,
        "general_advice": general_advice
    }

# ─── Module 4: AI-Powered Symptom Analyzer (Google Gemini) ──────────────

GEMINI_API_KEY = "AIzaSyDvr05CObK2VwSU9tTqVeN2h5a01faIG7c"

try:
    from google import genai as genai_new
    gemini_client = genai_new.Client(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini AI client initialized successfully!")
except Exception as e:
    gemini_client = None
    logger.error(f"❌ Error initializing Gemini: {e}")

MEDICAL_PROMPT = """You are a helpful medical assistant. A patient describes their symptoms below.
Analyze their symptoms and provide helpful medical advice.

Rules:
1. Identify the likely conditions (1-3 max)
2. For each condition, suggest a simple, commonly available medicine
3. Give the dosage and how long to take it
4. Rate urgency: "mild" (self-treatable), "moderate" (see doctor in 24h), or "severe" (go to hospital NOW)
5. Give one line of practical advice for each
6. If symptoms sound serious (chest pain, breathing difficulty, bleeding, fainting), ALWAYS mark as "severe" and tell them to see a doctor immediately
7. Keep language simple and friendly, like talking to a friend
8. IMPORTANT: Reply ONLY with valid JSON in this exact format, no extra text:

{{
  "conditions": [
    {{
      "symptom": "condition name",
      "medicine": "medicine name and dose",
      "timeline": "how often and how long",
      "urgency": "mild/moderate/severe",
      "advice": "one practical tip"
    }}
  ],
  "summary": "A short friendly 1-2 sentence summary of what they should do",
  "see_doctor": true/false
}}

Patient says: "{patient_text}"
"""


@app.post("/ai-analyze")
async def ai_analyze_symptoms(data: SymptomRequest):
    """Use Google Gemini AI to analyze symptoms and provide intelligent recommendations."""
    if gemini_client is None:
        raise HTTPException(status_code=500, detail="AI client is not initialized. Check your API key.")

    try:
        logger.info(f"AI analyzing: '{data.patient_text[:80]}...'")

        prompt = MEDICAL_PROMPT.replace("{patient_text}", data.patient_text)
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        # Extract the text and parse JSON
        raw_text = response.text.strip()
        
        # Remove markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]  # Remove first line
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3].strip()

        result = json.loads(raw_text)

        # Build response in the format frontend expects
        has_severe = any(c.get("urgency") == "severe" for c in result.get("conditions", []))
        
        recommendations = []
        detected_symptoms = []
        for condition in result.get("conditions", []):
            detected_symptoms.append(condition.get("symptom", "Unknown"))
            recommendations.append({
                "symptom": condition.get("symptom", "Unknown"),
                "medicine": condition.get("medicine", "Consult a doctor"),
                "timeline": condition.get("timeline", "As directed by doctor"),
                "urgency": condition.get("urgency", "moderate"),
                "advice": condition.get("advice", "")
            })

        logger.info(f"AI detected {len(detected_symptoms)} conditions: {detected_symptoms}")

        return {
            "detected_symptoms": detected_symptoms,
            "recommendations": recommendations,
            "has_severe": has_severe,
            "general_advice": result.get("summary", ""),
            "see_doctor": result.get("see_doctor", False),
            "ai_powered": True
        }

    except json.JSONDecodeError as e:
        logger.error(f"AI response parsing error: {e}")
        logger.error(f"Raw response: {raw_text[:200]}")
        raise HTTPException(status_code=500, detail="AI returned an unexpected format. Please try again.")
    except Exception as e:
        logger.error(f"AI analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)