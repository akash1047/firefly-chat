# Use an official Python 3.10 base image (compatible with TensorFlow)
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy your app code
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install tensorflow numpy pandas flask nltk scikit-learn

# Download NLTK data (optional, uncomment if needed)
# RUN python -m nltk.downloader punkt stopwords

# Expose the Flask port
EXPOSE 5000

# Run the application
CMD ["python", "chatbot.py"]
