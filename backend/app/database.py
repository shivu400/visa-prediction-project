import sqlite3

# Connect to the SQLite database (it will be created if it doesn't exist)
conn = sqlite3.connect('visa_predictions.db')
cursor = conn.cursor()

# Create the predictions table
cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT,
    age INTEGER,
    nationality TEXT,
    visa_type TEXT,
    destination_country TEXT,
    monthly_income_usd INTEGER,
    bank_balance_usd INTEGER,
    prev_visa_rejections INTEGER,
    has_criminal_record INTEGER,
    prediction_label TEXT,
    approval_probability REAL,
    risk_assessment TEXT,
    pdf_path TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database 'visa_predictions.db' and table 'predictions' created successfully.")