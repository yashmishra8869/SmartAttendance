# Use Python 3.10 to ensure compatibility with dependencies
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SMARTATTENDANCE_DATA_DIR=/data

# System packages
# - libgl1, libglib2.0-0: needed by OpenCV wheels at runtime
# - build-essential, cmake, pkg-config: needed to build dlib (dependency of face_recognition)
# - libopenblas-dev, liblapack-dev, libx11-dev: math/X11 deps for dlib
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 libglib2.0-0 \
       build-essential cmake pkg-config \
       libopenblas-dev liblapack-dev libx11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install dependencies first (better caching)
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the app source
COPY . .

# Create data dir
RUN mkdir -p /data \
    && chown -R root:root /data

# Default port; platform will override via PORT env
ENV PORT=8000
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["sh", "-c", "uvicorn web.main:app --host 0.0.0.0 --port ${PORT}"]
