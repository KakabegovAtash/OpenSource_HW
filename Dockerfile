# Use the official lightweight Python image
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model download script and download models during build
# This bakes the models into the Docker image so it starts up instantly
# without needing internet access at runtime.
COPY download_models.py .
RUN python download_models.py

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Create a non-root user to run the app for security
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Start the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
