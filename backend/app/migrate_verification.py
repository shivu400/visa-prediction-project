import sqlite3

conn = sqlite3.connect('visa_predictions.db')
cursor = conn.cursor()

try:
    # Add a column to track if the record has been verified (0 = False, 1 = True)
    cursor.execute("ALTER TABLE predictions ADD COLUMN is_verified INTEGER DEFAULT 0")
    
    # Add a column to store the admin-verified status ('Approved' or 'Rejected')
    cursor.execute("ALTER TABLE predictions ADD COLUMN verified_status TEXT DEFAULT NULL")
    
    print("Successfully added 'is_verified' and 'verified_status' columns.")
except sqlite3.OperationalError as e:
    # This will happen if the columns already exist, which is fine.
    print(f"Could not add columns, they might already exist: {e}")

conn.commit()
conn.close()