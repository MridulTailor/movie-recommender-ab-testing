# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if any)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Install pipenv
RUN pip install pipenv

# Copy the Pipfile and Pipfile.lock into the container
COPY Pipfile Pipfile.lock /app/

# Install dependencies via pipenv
# --system flag installs packages into the system python, avoiding the need for a virtualenv in the container
RUN pipenv sync --system

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "src/app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
