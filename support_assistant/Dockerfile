# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MOCK_LLM=1
ENV PYTHONPATH="/app:/app/support_assistant"

# Install system dependencies and curl for health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for optimal Docker layer caching
COPY requirements.txt .

# Upgrade pip and install Python requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire project directory into /app
COPY . .

# Switch working directory to support_assistant so local imports and paths resolve correctly
WORKDIR /app/support_assistant

# Ingest policy docs to generate the local ChromaDB vector store
RUN python ingest_docs.py

# Expose FastAPI port
EXPOSE 7860

# Command to run the Support Assistant FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]