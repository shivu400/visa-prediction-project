import sqlite3

conn = sqlite3.connect('visa_predictions.db')
cursor = conn.cursor()

try:
    # Add an 'is_admin' column to the users table, defaulting to 0 (False)
    cursor.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
    print("Successfully added 'is_admin' column to 'users' table.")
except sqlite3.OperationalError as e:
    # This will happen if the column already exists, which is fine.
    print(f"Could not add column, it might already exist: {e}")

conn.commit()
conn.close()