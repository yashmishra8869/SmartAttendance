# SmartAttendance - Web App

Run the FastAPI web app
- pip install -r requirements.txt
- uvicorn web.main:app --reload

Features
- Dashboard: monthly attendance percentage per student.
- Students: list, upload images to register, or register via webcam.
- Scan: webcam-based attendance marking.

Data storage
- Uses `SMARTATTENDANCE_DATA_DIR` if set, else the project root.
- Files: `encodings.pkl` and `attendance.csv` live in that directory.
- Set the env var to share data between CLI and web, e.g. on Windows PowerShell:
  `$env:SMARTATTENDANCE_DATA_DIR="D:\\SmartAttendance\\data"`

Docker
- Build: `docker build -t smart-attendance .`
- Run (persists data to host):
  `docker run -p 8000:8000 -e SMARTATTENDANCE_DATA_DIR=/data -v %cd%/data:/data smart-attendance`
  On Linux/macOS replace `%cd%` with `$(pwd)`.
- Open http://localhost:8000

Production tips
- Run a single worker (default) to avoid concurrent writes to `encodings.pkl`.
- Ensure camera access is allowed when using the Scan or Webcam Register pages.
- If dlib fails on some frames, app falls back to OpenCV Haar detection.
