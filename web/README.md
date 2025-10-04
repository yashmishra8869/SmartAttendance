# SmartAttendance - Web App

Run the FastAPI web app
- pip install -r requirements.txt
- uvicorn web.main:app --reload

Features
- Dashboard: monthly attendance percentage per student.
- Students: list, upload images to register, or register via webcam.
- Scan: webcam-based attendance marking.

Notes
- Uses encodings.pkl and attendance.csv at the project root by default.
- Allow camera access in the browser for Scan and Webcam Register pages.
