# AI-based Smart Attendance System

A complete, ready-to-run Python project that automates attendance tracking using AI-powered facial recognition with OpenCV and the `face_recognition` library.

Features:
- Register students via webcam or existing JPG/PNG images
- Real-time face detection and recognition from webcam
- Draw bounding boxes and names on live video
- Log attendance to `attendance.csv` with Name, Date, Time
- Ensure each student is logged once per day
- Self-contained, modular, and well-commented

## Requirements
- Python 3.8+
- OS: Windows, macOS, or Linux
- A working webcam (for registration and recognition from camera)

Python packages (installed via `pip`):
- opencv-python
- face_recognition (requires dlib; prebuilt wheels are available for most platforms)
- numpy
- pandas (used for efficient daily attendance checks; CSV fallback is implemented)

## Quick Start
1) Create a virtual environment (recommended) and install dependencies:

   pip install -r requirements.txt

2) Register students
- From webcam (capture 15 samples):

   python register_student.py --name "Alice Johnson" --from-webcam --samples 15

- From a directory of images:

   python register_student.py --name "Bob Smith" --images-dir ./images/bob

- From selected images:

   python register_student.py --name "Charlie" --images img1.jpg img2.png

3) Run real-time recognition and logging

   python recognize_and_log.py

- Options: `--camera-index 1` selects a different webcam. `--tolerance 0.5` makes matching stricter. Use `--headless` to disable video window.

4) View attendance logs
- Check `attendance.csv` for entries with headers: Name, Date, Time.

## Files
- `register_student.py` — Register students and build `encodings.pkl`.
- `recognize_and_log.py` — Recognize faces in real time and log attendance to `attendance.csv`.
- `encodings.pkl` — Auto-created face embeddings database.
- `attendance.csv` — Auto-created attendance log with headers.
- `requirements.txt` — Dependencies list.

## How It Works
1. Registration
- Captures multiple facial encodings per student (from webcam or images).
- Stores encodings and aligned names in `encodings.pkl`.
- Supports duplicate handling using `--append` and `--replace`.

2. Recognition
- Loads known encodings and names.
- For each video frame, detects faces and computes encodings.
- Matches encodings against known ones using Euclidean distance via `face_recognition.compare_faces`.
- On the first match per day, writes to `attendance.csv`.

## Tips and Troubleshooting
- If recognition is inconsistent, collect more samples per student (15–30 is typical).
- Good lighting and frontal face alignment improve accuracy.
- If `face_recognition` installation fails, install a compatible `dlib` wheel for your Python version and OS.
- To reset the database, delete `encodings.pkl`. To clear attendance, delete `attendance.csv`.

## Security and Privacy
- Encodings are not raw images; they are 128-d vectors. Still, handle data responsibly.
- Store files securely and obtain consent from participants.

## License
This sample is provided for educational purposes. Review local laws and institutional policies before deploying.
