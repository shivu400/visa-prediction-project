FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy all necessary app data ---

# Copy the models to /code/models
COPY models/ ./models/

# ⭐️⭐️⭐️ THE FIX ⭐️⭐️⭐️
# Instead of copying a folder that isn't in Git,
# just create an empty one inside the image.
RUN mkdir -p uploads

# Copy your backend application code (app/, etc.)
COPY backend/ .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]