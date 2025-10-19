import sqlite3
import uuid
import json
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import re
import fitz
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from fastapi.staticfiles import StaticFiles

# --- App Initialization & CORS ---
app = FastAPI(title="Visa Prediction API")
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Smart Model Loading from Registry ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
REGISTRY_PATH = os.path.join(MODELS_DIR, 'model_registry.json')

try:
    with open(REGISTRY_PATH, 'r') as f:
        model_registry = json.load(f)
    # Find the best model based on accuracy
    best_model_info = max(model_registry, key=lambda x: x['accuracy'])
    MODEL_PATH = os.path.join(MODELS_DIR, best_model_info['filename'])
    ML_MODEL_ACCURACY = best_model_info['accuracy'] * 100
    print(f"--- Loading best model: {best_model_info['name']} with accuracy {ML_MODEL_ACCURACY:.2f}% ---")
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"--- ERROR: model_registry.json not found at {REGISTRY_PATH}. Please run the training script. ---")
    exit() # Exit if no registry is found

# --- DB Path & Static Files ---
DB_PATH = os.path.join(os.path.dirname(__file__), 'visa_predictions.db')
uploads_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

# --- Pydantic Data Model ---
class VisaApplication(BaseModel):
    full_name: str
    age: int
    nationality: str
    marital_status: str
    education_level: str
    destination_country: str
    visa_type: str
    duration_of_stay: int
    monthly_income_usd: int
    bank_balance_usd: int
    prev_countries_visited: int
    prev_visa_rejections: int
    has_return_ticket: int
    has_criminal_record: int
    pdf_filename: Optional[str] = None

# --- Helper Functions and Rules ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

VISA_RULES = {
    "Tourist": {"min_bank_balance": 5000},
    "Student": {"min_bank_balance": 10000},
    "Work": {"min_bank_balance": 2000}
}

# --- API Endpoints ---
@app.post("/predict")
def predict_visa_approval(application: VisaApplication):
    model_input_data = application.model_dump(exclude={'pdf_filename', 'full_name'})
    input_data = pd.DataFrame([model_input_data])
    
    prediction = model.predict(input_data)[0]
    confidence_score = model.predict_proba(input_data)[0]
    approval_probability = confidence_score[1]

    risk_level = "low"
    if approval_probability < 0.4: risk_level = "high"
    elif approval_probability < 0.7: risk_level = "medium"
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO predictions (full_name, age, nationality, visa_type, destination_country, monthly_income_usd, bank_balance_usd, prev_visa_rejections, has_criminal_record, prediction_label, approval_probability, risk_assessment, pdf_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (application.full_name, application.age, application.nationality, application.visa_type, application.destination_country, application.monthly_income_usd, application.bank_balance_usd, application.prev_visa_rejections, application.has_criminal_record, "Approved" if prediction == 1 else "Rejected", float(approval_probability), risk_level, application.pdf_filename)
    )
    conn.commit()
    conn.close()

    concerns = []
    rules = VISA_RULES.get(application.visa_type, VISA_RULES.get("Tourist", {}))
    if application.bank_balance_usd < rules.get("min_bank_balance", 0):
        concerns.append(f"Bank balance is below the recommended ${rules.get('min_bank_balance', 0)} for a {application.visa_type} visa.")

    result = {
        "prediction_label": "Approved" if prediction == 1 else "Rejected",
        "approval_probability": float(approval_probability),
        "model_confidence": float(confidence_score[prediction]),
        "risk_assessment": risk_level,
        "areas_of_concern": concerns,
        "applicant_info": application.model_dump()
    }
    return result

@app.get("/history")
def get_history():
    conn = get_db_connection()
    predictions = conn.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10').fetchall()
    conn.close()
    return [dict(row) for row in predictions]

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(uploads_dir, unique_filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    extracted_data = {}
    try:
        pdf_document = fitz.open(file_path)
        full_text = ""
        for page in pdf_document:
            full_text += page.get_text()
        
        age_match = re.search(r"Age:\s*(\d+)", full_text, re.IGNORECASE)
        if age_match:
            extracted_data['age'] = int(age_match.group(1))
    except Exception as e:
        return {"error": f"Failed to process PDF: {str(e)}"}
    
    return {"extracted_data": extracted_data, "saved_filename": unique_filename}
@app.get("/admin/stats")
def get_admin_stats():
    """
    Returns aggregated statistics for the admin dashboard.
    """
    conn = get_db_connection()
    total_predictions = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    approved_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction_label = 'Approved'").fetchone()[0]
    high_risk_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_assessment = 'high'").fetchone()[0]
    conn.close()
    
    approval_rate = (approved_count / total_predictions) * 100 if total_predictions > 0 else 0
    
    return {
        "total_predictions": total_predictions,
        "approval_rate": approval_rate,
        "high_risk_cases": high_risk_count # ⭐️ FIX: Use the correct variable name here
    }
@app.get("/admin/all-predictions")
def get_all_predictions():
    conn = get_db_connection()
    predictions = conn.execute('SELECT * FROM predictions ORDER BY timestamp DESC').fetchall()
    conn.close()
    return [dict(row) for row in predictions]

@app.get("/insights")
def get_insights():
    conn = get_db_connection()
    predictions_today = conn.execute("SELECT COUNT(*) FROM predictions WHERE DATE(timestamp) = DATE('now', 'localtime')").fetchone()[0]
    conn.close()
    return {"ml_accuracy": ML_MODEL_ACCURACY, "predictions_today": predictions_today}
    # ... (keep all existing code and endpoints)

# --- ⭐️ NEW: Endpoint to fetch all prediction history ---
@app.get("/user/all-predictions")
def get_user_all_predictions():
    """
    Returns all prediction records from the database for the history page.
    """
    conn = get_db_connection()
    predictions = conn.execute('SELECT * FROM predictions ORDER BY timestamp DESC').fetchall()
    conn.close()
    return [dict(row) for row in predictions]
    # ... (keep all existing code and endpoints)
@app.get("/insights")
def get_insights():
    """
    Returns AI insights for the sidebar.
    """
    conn = get_db_connection()
    # Query for predictions made today
    predictions_today = conn.execute("SELECT COUNT(*) FROM predictions WHERE DATE(timestamp) = DATE('now', 'localtime')").fetchone()[0]
    conn.close()
    
    return {
        "ml_accuracy": ML_MODEL_ACCURACY,
        "predictions_today": predictions_today
    }
# --- ⭐️ NEW: Endpoint for dashboard country chart ---
@app.get("/dashboard/country-analytics")
def get_country_analytics():
    """
    Returns aggregated data on predictions by country.
    """
    conn = get_db_connection()
    # This query counts approved/rejected for each country
    query = """
        SELECT
            destination_country,
            SUM(CASE WHEN prediction_label = 'Approved' THEN 1 ELSE 0 END) as approved,
            SUM(CASE WHEN prediction_label = 'Rejected' THEN 1 ELSE 0 END) as rejected
        FROM predictions
        GROUP BY destination_country
        ORDER BY (approved + rejected) DESC
        LIMIT 5;
    """
    data = conn.execute(query).fetchall()
    conn.close()
    return [dict(row) for row in data]
    # ... (at the end of your file, before the VISA_RULES)

