import pandas as pd
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# --- Load Data ---
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'visa_applications.csv')
df = pd.read_csv(DATA_PATH)

# --- Preprocessing Setup ---
X = df.drop('visa_approved', axis=1)
y = df['visa_approved']
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Define Models to Train ---
models_to_train = {
    "random_forest_v1": RandomForestClassifier(random_state=42),
    "xgboost_v1": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "lightgbm_v1": LGBMClassifier(random_state=42)
}

# --- Train, Evaluate, and Save ---
model_registry = []
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

for name, model in models_to_train.items():
    print(f"--- Training {name} ---")
    
    # Create a pipeline with the preprocessor and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {name}: {accuracy:.4f}")
    
    # Save the model
    model_filename = f"{name}.joblib"
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    
    # Add to our registry
    model_registry.append({
        "name": name,
        "filename": model_filename,
        "accuracy": accuracy,
        "algorithm": model.__class__.__name__,
        "timestamp": pd.Timestamp.now().isoformat()
    })

# --- Save the Model Registry ---
REGISTRY_PATH = os.path.join(MODELS_DIR, 'model_registry.json')
with open(REGISTRY_PATH, 'w') as f:
    json.dump(model_registry, f, indent=4)

print("\n--- Model training complete. Registry created at model_registry.json ---")