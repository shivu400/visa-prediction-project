import sqlite3

conn = sqlite3.connect('visa_predictions.db')
cursor = conn.cursor()

try:
    print("Renaming currency and duration columns...")
    cursor.execute("ALTER TABLE predictions RENAME COLUMN monthly_income_usd TO monthly_income_inr")
    cursor.execute("ALTER TABLE predictions RENAME COLUMN bank_balance_usd TO bank_balance_inr")
    cursor.execute("ALTER TABLE predictions RENAME COLUMN duration_of_stay TO duration_of_stay_months")
    print("Columns renamed successfully.")
except sqlite3.OperationalError as e:
    print(f"Could not rename columns, they might already be renamed or another error occurred: {e}")

conn.commit()
conn.close()