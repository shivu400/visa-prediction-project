FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Create the empty uploads folder ---
RUN mkdir -p uploads

# --- Copy your ENTIRE backend application ---
# This one command copies app/, models/, and everything else
# from backend/ into the /code/ directory.
COPY backend/ .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]