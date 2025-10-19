import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib # <-- IMPORT JOBLIB
import os # <-- IMPORT OS

# --- Load the Dataset ---
# We adjust the path to go up one level from 'scripts' then into 'data'
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'visa_applications.csv')
df = pd.read_csv(DATA_PATH)

# --- Define Features (X) and Target (y) ---
X = df.drop('visa_approved', axis=1)
y = df['visa_approved']

categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# --- Create a Preprocessing Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Define the Model ---
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42))])

# --- Train the Model ---
print("Training the Random Forest model...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- Evaluate Model Performance ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- ⭐️ NEW: SAVE THE TRAINED MODEL ⭐️ ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True) # Create the models directory if it doesn't exist
MODEL_PATH = os.path.join(MODEL_DIR, 'visa_predictor_model_v1.joblib')

print(f"\nSaving model to: {MODEL_PATH}")
joblib.dump(model, MODEL_PATH)
print("Model saved successfully!")