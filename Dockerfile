FROM python:3.12-slim

LABEL maintainer="Cash Contributors"
LABEL description="Cash - Smart caching for Jupyter notebooks"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install cash with all extras
WORKDIR /app
COPY . /app/
RUN pip install --no-cache-dir -e ".[all,dev]"

# Install Jupyter
RUN pip install --no-cache-dir jupyterlab

# Expose Jupyter port
EXPOSE 8888

# Default command: start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
