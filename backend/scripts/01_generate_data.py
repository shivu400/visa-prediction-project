import pandas as pd
import numpy as np
from faker import Faker
import random
import os # <-- IMPORT OS

# --- Configuration ---
NUM_RECORDS = 10000

# --- Define Possible Values for Categorical Data ---
NATIONALITIES = ['Indian', 'American', 'British', 'Chinese', 'German', 'Canadian', 'Australian', 'Nigerian']
MARITAL_STATUSES = ['Single', 'Married', 'Divorced']
EDUCATION_LEVELS = ['High School', 'Bachelors', 'Masters', 'PhD']
DESTINATIONS = ['USA', 'UK', 'Canada', 'Germany', 'France', 'Australia']
VISA_TYPES = ['Tourist', 'Student', 'Work']

# --- Create the Dataset ---
data = []

for _ in range(NUM_RECORDS):
    # (The data generation logic is the same, so it is omitted here for brevity)
    age = np.random.randint(18, 70)
    nationality = random.choice(NATIONALITIES)
    marital_status = random.choice(MARITAL_STATUSES)
    education_level = random.choice(EDUCATION_LEVELS)
    destination_country = random.choice(DESTINATIONS)
    visa_type = random.choice(VISA_TYPES)
    duration_of_stay = np.random.randint(7, 365)
    monthly_income_usd = np.random.randint(500, 15001)
    bank_balance_usd = np.random.randint(1000, 200001)
    prev_countries_visited = np.random.randint(0, 21)
    prev_visa_rejections = np.random.randint(0, 6)
    has_return_ticket = random.choice([0, 1])
    has_criminal_record = np.random.choice([0, 1], p=[0.95, 0.05])
    
    approval_score = 0
    if monthly_income_usd > 4000: approval_score += 2
    if bank_balance_usd > 50000: approval_score += 3
    if education_level in ['Masters', 'PhD']: approval_score += 2
    if prev_countries_visited > 5: approval_score += 1
    if has_return_ticket == 1: approval_score += 1
    if prev_visa_rejections > 0: approval_score -= (prev_visa_rejections * 3)
    if has_criminal_record == 1: approval_score -= 5
    if monthly_income_usd < 1000: approval_score -= 2
    if bank_balance_usd < 5000: approval_score -= 2
    
    prob_approval = 1 / (1 + np.exp(-approval_score))
    visa_approved = 1 if np.random.rand() < prob_approval else 0

    data.append([
        age, nationality, marital_status, education_level,
        destination_country, visa_type, duration_of_stay,
        monthly_income_usd, bank_balance_usd, prev_countries_visited,
        prev_visa_rejections, has_return_ticket, has_criminal_record,
        visa_approved
    ])

# --- Create a Pandas DataFrame ---
column_names = [
    'age', 'nationality', 'marital_status', 'education_level',
    'destination_country', 'visa_type', 'duration_of_stay',
    'monthly_income_usd', 'bank_balance_usd', 'prev_countries_visited',
    'prev_visa_rejections', 'has_return_ticket', 'has_criminal_record',
    'visa_approved'
]
df = pd.DataFrame(data, columns=column_names)

# --- ⭐️ NEW: SAVE THE CSV TO THE CORRECT 'data' FOLDER ⭐️ ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True) # Create the data directory if it doesn't exist
DATA_PATH = os.path.join(DATA_DIR, 'visa_applications.csv')

print(f"Saving data to: {DATA_PATH}")
df.to_csv(DATA_PATH, index=False)
print("Data saved successfully!")