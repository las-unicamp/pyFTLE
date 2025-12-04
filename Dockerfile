FROM python:3.10-slim

LABEL maintainer="lasunicamp"
LABEL description="Container with pyFTLE installed and ready to use."

# Install system dependencies for PyVista/VTK
RUN apt-get update && apt-get install -y \
    libgl1 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m pyuser
USER pyuser
WORKDIR /home/pyuser

# Install package directly from PyPI
RUN pip install --user --no-cache-dir pyftle

# Add user installation path into PATH
ENV PATH="/home/pyuser/.local/bin:${PATH}"

# Entrypoint and default command
ENTRYPOINT ["pyftle"]
CMD ["--help"]
