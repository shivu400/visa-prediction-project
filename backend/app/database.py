import sqlite3

conn = sqlite3.connect('visa_predictions.db')
cursor = conn.cursor()

# Create the predictions table with INR and Months columns
cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT, age INTEGER, nationality TEXT, visa_type TEXT,
    destination_country TEXT,
    monthly_income_inr INTEGER,      -- Renamed
    bank_balance_inr INTEGER,        -- Renamed
    prev_visa_rejections INTEGER, has_criminal_record INTEGER,
    prediction_label TEXT, approval_probability REAL, risk_assessment TEXT,
    pdf_path TEXT,
    duration_of_stay_months INTEGER, -- Renamed
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()
conn.close()

print("Database 'visa_predictions.db' and table 'predictions' (with INR/Months) created successfully.")