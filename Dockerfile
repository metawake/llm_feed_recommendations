# Use a lightweight Python base image
FROM python:3.10-slim

# Install SQLite3 and other required packages
RUN apt-get update && apt-get install -y \
    sqlite3 \
    libsqlite3-dev \
 && rm -rf /var/lib/apt/lists/*

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent Python output from being buffered
ENV PYTHONUNBUFFERED=1

# Create a working directory
WORKDIR /app

# Copy requirements first to leverage Docker’s layer caching
COPY requirements.txt /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose the FastAPI port
EXPOSE 8000

# Run the app with uvicorn
CMD ["uvicorn", "rss:app", "--host", "0.0.0.0", "--port", "8000"]
