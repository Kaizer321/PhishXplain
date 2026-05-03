from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import uvicorn
import csv
import os
import time
from feature_extractor import FeatureExtractor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PhishXplain API", description="Explainable Phishing Detection", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

# Load models at startup
logger.info("Loading models...")
try:
    model = joblib.load('models/xgboost_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    top_features = joblib.load('models/top_14_features.joblib')
    
    extractor = FeatureExtractor(top_features)
    explainer = shap.TreeExplainer(model)
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    # Initialize as None for local testing if models aren't ready
    model, scaler, top_features, extractor, explainer = None, None, None, None, None

class URLRequest(BaseModel):
    url: str

class FeedbackRequest(BaseModel):
    url: str
    is_phishing: bool
    notes: str = ""

@app.post("/predict")
def predict_url(request: URLRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Models not loaded")
        
    start_time = time.time()
    url = request.url
    
    try:
        # 1. Extract features
        raw_features = extractor.get_feature_vector(url)
        df_features = pd.DataFrame([raw_features], columns=top_features)
        
        # 2. Normalize
        scaled_features = pd.DataFrame(scaler.transform(df_features), columns=top_features)
        
        # 3. Predict
        prob = model.predict_proba(scaled_features)[0][1]
        prediction = int(model.predict(scaled_features)[0])
        
        # 4. Explain with SHAP
        shap_values = explainer(scaled_features)
        # Convert SHAP values to a dictionary of {feature: contribution}
        # In newer shap versions, shap_values is an Explanation object
        if hasattr(shap_values, 'values'):
            contributions = dict(zip(top_features, shap_values.values[0].tolist()))
            base_value = float(shap_values.base_values[0])
        else:
            # Fallback for older SHAP versions
            contributions = dict(zip(top_features, shap_values[0].tolist()))
            base_value = float(explainer.expected_value)
            
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "url": url,
            "phishing_probability": float(prob),
            "is_phishing": bool(prediction),
            "explanation": {
                "base_value": base_value,
                "feature_contributions": contributions
            },
            "latency_ms": round(latency_ms, 2)
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    """Continuous Learning Loop Endpoint"""
    feedback_file = 'feedback_log.csv'
    file_exists = os.path.isfile(feedback_file)
    
    with open(feedback_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'url', 'is_phishing', 'notes'])
        writer.writerow([time.time(), request.url, request.is_phishing, request.notes])
        
    return {"status": "success", "message": "Feedback recorded for next training cycle."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
