# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependency management files
COPY pyproject.toml uv.lock ./

# Install any needed packages specified in pyproject.toml
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache --no-deps .

# Copy the rest of the application's code
COPY . .

# Define the entrypoint for the container
ENTRYPOINT ["pyftle"]

