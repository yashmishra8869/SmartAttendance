# SmartAttendance - Web App

Run the FastAPI web app
- pip install -r requirements.txt
- uvicorn web.main:app --reload

Features
- Dashboard: monthly attendance percentage per student.
- Students: list, upload images to register, or register via webcam.
- Scan: webcam-based attendance marking.

Notes
- Uses encodings.pkl and attendance.csv at project root.
- Single-process recommended (one uvicorn worker) to avoid concurrent pickle writes.
- If dlib errors occur on some frames, app falls back to OpenCV Haar for detection.
