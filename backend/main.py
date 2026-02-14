from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from datetime import datetime
import logging
import sys

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

if __name__ == "__main__":
    import uvicorn
    # Verify port availability before running could be added here, 
    # but uvicorn handles it by crashing if port is taken.
    uvicorn.run(app, host="0.0.0.0", port=8000)