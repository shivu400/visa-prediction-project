# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file from the root
# (This path is relative to the build context, which is the root)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your backend application code into the container
COPY backend/ .

# Expose the port your app runs on
EXPOSE 8000

# Command to run your FastAPI app
# This assumes your FastAPI 'app' object is in 'backend/app/main.py'
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]