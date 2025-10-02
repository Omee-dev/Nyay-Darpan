# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (optional, depending on OCR libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt punkt_tab

# Copy the rest of your app
COPY . .

# Expose Flask port
EXPOSE 5000

# Run your app
CMD ["python", "Decision_Assist.py"]
