# Use Python 3.10 to ensure compatibility with dependencies
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SMARTATTENDANCE_DATA_DIR=/data

# System packages
# - libgl1, libglib2.0-0: needed by OpenCV wheels at runtime
# - libx11-6: runtime dependency for dlib/OpenCV stack
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 libglib2.0-0 \
         libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install dependencies first (better caching)
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install --only-binary=:all: dlib-bin==19.24.6 \
    && pip install face-recognition-models==0.3.0 \
    && pip install face_recognition==1.3.0 --no-deps \
    && pip install -r requirements.txt

# Copy the app source
COPY . .

# Create data dir
RUN mkdir -p /data \
    && chown -R root:root /data

# Default port aligned with Render's web service default.
ENV PORT=10000
EXPOSE 10000

# Start FastAPI with Uvicorn
CMD ["sh", "-c", "uvicorn web.main:app --host 0.0.0.0 --port ${PORT}"]
