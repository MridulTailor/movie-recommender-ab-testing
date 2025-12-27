# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for compilation usually)
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "src/app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
