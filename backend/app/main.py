# --- All Imports Moved to Top ---
import sqlite3
import uuid
import json
import os
import re
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import joblib
import pandas as pd
import shap
import fitz  # PyMuPDF

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

# --- Load Environment Variables AFTER imports ---
load_dotenv()


# --- App Initialization, CORS ---
app = FastAPI(title="Visa Prediction API")
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Smart Model & SHAP Explainer Loading ---
try:
    # ⭐️ FIX: Path from app/main.py -> backend/models/
    MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    REGISTRY_PATH = os.path.join(MODELS_DIR, 'model_registry.json')

    with open(REGISTRY_PATH, 'r') as f:
        model_registry = json.load(f)
    best_model_info = max(model_registry, key=lambda x: x['accuracy'])
    MODEL_PATH = os.path.join(MODELS_DIR, best_model_info['filename'])
    ML_MODEL_ACCURACY = best_model_info['accuracy'] * 100
    print(f"--- Loading best model: {best_model_info['name']} with accuracy {ML_MODEL_ACCURACY:.2f}% ---")
    
    model = joblib.load(MODEL_PATH)
    classifier = model[-1] 
    
    print("--- Initializing SHAP TreeExplainer... ---")
    explainer = shap.TreeExplainer(classifier)
    print("--- SHAP Explainer loaded successfully. ---")

except Exception as e:
    print(f"--- FATAL ERROR: Model or SHAP Explainer failed to load: {e} ---")
    print("--- Please ensure 'model_registry.json' is correct and 'shap' is installed. ---")
    model = None
    explainer = None

# --- DB Path & Static Files ---
# ⭐️ FIX: Path from app/main.py -> backend/visa_predictions.db
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'visa_predictions.db')
# ⭐️ FIX: Path from app/main.py -> root/uploads/
uploads_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'uploads') 
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

# --- Pydantic Data Models ---
class VisaApplication(BaseModel):
    full_name: str
    age: int
    nationality: str
    marital_status: str
    education_level: str
    destination_country: str
    visa_type: str
    duration_of_stay_months: int
    monthly_income_inr: int
    bank_balance_inr: int
    prev_countries_visited: int
    prev_visa_rejections: int
    has_return_ticket: int
    has_criminal_record: int
    pdf_filename: Optional[str] = None

class VerificationUpdate(BaseModel):
    verified_status: str

# --- Helper Functions and Rules (INR) ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

APPROX_INR_PER_USD = 80
VISA_RULES = {
    "Tourist": {"min_bank_balance": 5000 * APPROX_INR_PER_USD},
    "Student": {"min_bank_balance": 10000 * APPROX_INR_PER_USD},
    "Work": {"min_bank_balance": 2000 * APPROX_INR_PER_USD}
}

# --- API Endpoints ---
@app.post("/predict")
def predict_visa_approval(application: VisaApplication):
    if not model or not explainer:
        raise HTTPException(status_code=500, detail="Model or SHAP Explainer is not loaded. Check server logs.")

    # --- 1. PREPARE DATA ---
    model_input_dict = application.model_dump(exclude={'pdf_filename', 'full_name'})
    model_input_dict['monthly_income_usd'] = round(application.monthly_income_inr / APPROX_INR_PER_USD)
    model_input_dict['bank_balance_usd'] = round(application.bank_balance_inr / APPROX_INR_PER_USD)
    model_input_dict['duration_of_stay'] = application.duration_of_stay_months * 30
    
    if 'monthly_income_inr' in model_input_dict:
        del model_input_dict['monthly_income_inr']
    if 'bank_balance_inr' in model_input_dict:
        del model_input_dict['bank_balance_inr']
    if 'duration_of_stay_months' in model_input_dict:
        del model_input_dict['duration_of_stay_months']
    
    input_data = pd.DataFrame([model_input_dict])

    # --- 2. GET PREDICTION ---
    prediction = model.predict(input_data)[0]
    confidence_score = model.predict_proba(input_data)[0]
    approval_probability = confidence_score[1]
    risk_level = "low"
    
    if approval_probability < 0.4:
        risk_level = "high"
    elif approval_probability < 0.7:
        risk_level = "medium"

    # --- 3. GET SHAP EXPLANATION ---
    feature_importance = []
    try:
        preprocessor = model[:-1]
        processed_input = preprocessor.transform(input_data)
        feature_names = preprocessor.get_feature_names_out()
        shap_values = explainer.shap_values(processed_input)
        
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_for_approval = shap_values[1][0]
        else:
            shap_values_for_approval = shap_values[0] 

        contributions = dict(zip(feature_names, shap_values_for_approval))
        sorted_contributions = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
        
        feature_importance = [
            {"feature": feature.replace('_', ' ').title(), "impact": impact}
            for feature, impact in sorted_contributions
        ]
        
    except Exception as e:
        print(f"--- SHAP Error: {e} ---")
        feature_importance = [{"feature": "Explanation Error", "impact": 0}]

    # --- 4. SAVE TO DATABASE ---
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # ⭐️ TYPO FIX: Removed extra dot from 'cursor..execute'
    cursor.execute(
        '''INSERT INTO predictions (
            full_name, age, nationality, visa_type, destination_country,
            monthly_income_inr, bank_balance_inr, prev_visa_rejections,
            has_criminal_record, prediction_label, approval_probability,
            risk_assessment, pdf_path, duration_of_stay_months, is_verified, verified_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)''',
        (
            application.full_name, application.age, application.nationality,
            application.visa_type, application.destination_country,
            application.monthly_income_inr, application.bank_balance_inr,
            application.prev_visa_rejections, application.has_criminal_record,
            "Approved" if prediction == 1 else "Rejected",
            float(approval_probability), risk_level, application.pdf_filename,
            application.duration_of_stay_months
        )
    )
    conn.commit()
    conn.close()

    # --- 5. GENERATE CONCERNS & FINAL RESPONSE ---
    concerns = []
    rules = VISA_RULES.get(application.visa_type, VISA_RULES.get("Tourist", {}))
    if application.bank_balance_inr < rules.get("min_bank_balance", 0):
        concerns.append(f"Bank balance is below the recommended INR {rules.get('min_bank_balance', 0):,} for a {application.visa_type} visa.")
    
    result = {
        "prediction_label": "Approved" if prediction == 1 else "Rejected",
        "approval_probability": float(approval_probability),
        "model_confidence": float(confidence_score[prediction]),
        "risk_assessment": risk_level,
        "areas_of_concern": concerns,
        "applicant_info": application.model_dump(),
        "feature_importance": feature_importance
    }
    return result

@app.get("/history")
def get_history():
    conn = get_db_connection()
    predictions = conn.execute('SELECT id, full_name, nationality, visa_type, destination_country, prediction_label, approval_probability, risk_assessment, timestamp, duration_of_stay_months FROM predictions ORDER BY timestamp DESC LIMIT 10').fetchall()
    conn.close()
    return [dict(row) for row in predictions]

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(uploads_dir, unique_filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    extracted_data = {}
    FIELD_NAME_MAP = {
        'FullName': 'full_name', 'Applicant_Name': 'full_name', 'Age': 'age',
        'Nationality': 'nationality', 'MaritalStatus': 'marital_status',
        'Destination_Country': 'destination_country', 'VisaType': 'visa_type',
        'StayDuration_Months': 'duration_of_stay_months', 'Income_INR': 'monthly_income_inr',
        'BankBalance_INR': 'bank_balance_inr',
    }
    
    try:
        pdf_document = fitz.open(file_path)
        for page in pdf_document:
            for widget in page.widgets():
                if widget.field_name in FIELD_NAME_MAP:
                    our_key = FIELD_NAME_MAP[widget.field_name]
                    extracted_data[our_key] = widget.field_value
        
        if not extracted_data:
            print("--- No form fields found, falling back to Regex text search ---")
            full_text = ""
            for page in pdf_document:
                full_text += page.get_text()
            
            age_match = re.search(r"Age:\s*(\d+)", full_text, re.IGNORECASE)
            if age_match:
                extracted_data['age'] = int(age_match.group(1))
            
            income_match_inr = re.search(r"Monthly Income.*?₹\s*([\d,]+)", full_text, re.IGNORECASE)
            if income_match_inr:
                extracted_data['monthly_income_inr'] = int(income_match_inr.group(1).replace(',', ''))
            
            balance_match_inr = re.search(r"Bank Balance.*?₹\s*([\d,]+)", full_text, re.IGNORECASE)
            if balance_match_inr:
                extracted_data['bank_balance_inr'] = int(balance_match_inr.group(1).replace(',', ''))
            
            name_match = re.search(r"Full Name:\s*(.*)", full_text, re.IGNORECASE)
            if name_match:
                extracted_data['full_name'] = name_match.group(1).strip()
            
            nationality_match = re.search(r"Nationality:\s*(.*)", full_text, re.IGNORECASE)
            if nationality_match:
                extracted_data['nationality'] = nationality_match.group(1).strip()
            
            marital_match = re.search(r"Marital Status:\s*(.*)", full_text, re.IGNORECASE)
            if marital_match:
                extracted_data['marital_status'] = marital_match.group(1).strip()
            
        pdf_document.close()
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return {"error": f"Failed to process PDF: {str(e)}", "extracted_data": {}, "saved_filename": None}
    
    print(f"Extracted data: {extracted_data}")
    return {"extracted_data": extracted_data, "saved_filename": unique_filename}

@app.get("/admin/stats")
def get_admin_stats():
    conn = get_db_connection()
    total_predictions = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    approved_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction_label = 'Approved'").fetchone()[0]
    high_risk_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_assessment = 'high'").fetchone()[0]
    conn.close()
    approval_rate = (approved_count / total_predictions) * 100 if total_predictions > 0 else 0
    return {"total_predictions": total_predictions, "approval_rate": approval_rate, "high_risk_cases": high_risk_count}

@app.get("/admin/all-predictions")
def get_all_predictions():
    conn = get_db_connection()
    predictions = conn.execute(
        '''SELECT id, full_name, nationality, visa_type, destination_country, 
           prediction_label, approval_probability, risk_assessment, timestamp, 
           pdf_path, duration_of_stay_months,
           is_verified, verified_status
           FROM predictions ORDER BY timestamp DESC'''
    ).fetchall()
    conn.close()
    return [dict(row) for row in predictions]

@app.get("/insights")
def get_insights():
    conn = get_db_connection()
    predictions_today = conn.execute("SELECT COUNT(*) FROM predictions WHERE DATE(timestamp) = DATE('now', 'localtime')").fetchone()[0]
    conn.close()
    return {"ml_accuracy": ML_MODEL_ACCURACY, "predictions_today": predictions_today}

@app.get("/user/all-predictions")
def get_user_all_predictions():
    conn = get_db_connection()
    predictions = conn.execute('SELECT id, full_name, nationality, visa_type, destination_country, prediction_label, approval_probability, risk_assessment, timestamp, duration_of_stay_months FROM predictions ORDER BY timestamp DESC').fetchall()
    conn.close()
    return [dict(row) for row in predictions]

@app.get("/dashboard/country-analytics")
def get_country_analytics():
    conn = get_db_connection()
    query = """
        SELECT destination_country,
               SUM(CASE WHEN prediction_label = 'Approved' THEN 1 ELSE 0 END) as approved,
               SUM(CASE WHEN prediction_label = 'Rejected' THEN 1 ELSE 0 END) as rejected
        FROM predictions GROUP BY destination_country ORDER BY (approved + rejected) DESC LIMIT 5;
    """
    data = conn.execute(query).fetchall()
    conn.close()
    return [dict(row) for row in data]

@app.put("/admin/verify/{prediction_id}")
def verify_prediction(prediction_id: int, update: VerificationUpdate):
    if update.verified_status not in ["Approved", "Rejected"]:
        raise HTTPException(status_code=400, detail="Invalid verification status. Must be 'Approved' or 'Rejected'.")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    prediction = cursor.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,)).fetchone()
    
    if not prediction:
        conn.close()
        raise HTTPException(status_code=404, detail="Prediction not found")
        
    cursor.execute(
        '''UPDATE predictions 
           SET is_verified = 1, verified_status = ? 
           WHERE id = ?''',
        (update.verified_status, prediction_id)
    )
    conn.commit()
    updated_prediction = cursor.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,)).fetchone()
    conn.close()
    return dict(updated_prediction)

# --- Initialize Chatbot (RAG Pipeline) ---
qa_chain = None
try:
    print("--- Initializing AI Chatbot (RAG)... ---")
    
    openrouter_llm = ChatOpenAI(
        model="meta-llama/llama-4-scout:free",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        temperature=0.1
    )
    
    # ⭐️ FIX: Path from app/main.py -> backend/knowledge_base.txt
    KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'knowledge_base.txt')
    loader = TextLoader(KNOWLEDGE_BASE_PATH)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.environ.get("OPENROUTER_API_KEY")
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    # ⭐️ TYPO FIX: Changed 'as_ri()' to 'as_retriever()'
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=openrouter_llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )
    
    print("--- AI Chatbot loaded successfully. ---")

except ImportError:
    print("--- AI Chatbot libraries not found. Skipping chatbot initialization. ---")
except Exception as e:
    print(f"--- ERROR initializing AI Chatbot: {e} ---")
    print("--- Check if OPENROUTER_API_KEY is set correctly. ---")

# --- Chatbot Pydantic Models ---
class ChatQuery(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

# --- Chatbot API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def handle_chat_query(query: ChatQuery):
    if qa_chain is None:
        raise HTTPException(status_code=500, detail="Chatbot is not initialized. Check server logs.")
    
    try:
        result = qa_chain.invoke({"question": query.query, "chat_history": []})
        return ChatResponse(answer=result['answer'])
    except Exception as e:
        print(f"--- Chat Error: {e} ---")
        # ⭐️ TYPO FIX: Changed '50Do' to '500'
        raise HTTPException(status_code=500, detail="Error processing chat query.")