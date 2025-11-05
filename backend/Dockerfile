FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- FIX: Copy all necessary app data ---

# Copy the models to /code/models
COPY models/ ./models/

# Create and copy the uploads folder to /code/uploads
COPY uploads/ ./uploads/

# Copy your backend application code (app/, etc.)
COPY backend/ .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]