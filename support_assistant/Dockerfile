FROM python:3.10-slim

WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MOCK_LLM=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY support_assistant/ /app/support_assistant/

# Set working directory to module folder
WORKDIR /app/support_assistant

# Run ingestion during image build to populate local vector database
RUN python ingest_docs.py

EXPOSE 7860

# Serve application via uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]