# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Add the local bin directory to the PATH
ENV PATH="/root/.local/bin:${PATH}"
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/scripts"

# Copy the dependency management files
COPY pyproject.toml uv.lock README.md CMakeLists.txt ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential cmake git

# Install any needed packages specified in pyproject.toml
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache .

# Define the entrypoint for the container
ENTRYPOINT ["pyftle"]

