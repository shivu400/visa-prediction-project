import sqlite3

# Connect to the existing SQLite database
conn = sqlite3.connect('visa_predictions.db')
cursor = conn.cursor()

# Create the users table to store Google login information
# The 'IF NOT EXISTS' ensures we don't accidentally try to create it more than once
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    google_id TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    name TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Authentication setup complete: 'users' table has been added to the database.")