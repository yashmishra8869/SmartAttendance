#!/usr/bin/env python3
"""
AI-based Smart Attendance System - Real-time Recognition and Attendance Logging

Usage examples:
1) Default webcam, default tolerance:
   python recognize_and_log.py

2) Specify camera index and tolerance:
   python recognize_and_log.py --camera-index 1 --tolerance 0.5

3) Headless run (no GUI window), useful for servers:
   python recognize_and_log.py --headless

4) Force a specific video backend (Windows often prefers dshow over ms-mf):
   python recognize_and_log.py --backend dshow

Notes:
- This script loads face encodings from encodings.pkl.
- When a known face is recognized, it logs Name, Date (YYYY-MM-DD), Time (HH:MM:SS) into attendance.csv.
- Each student is logged at most once per day.
- Press 'q' to quit.
"""

import argparse
import os
import csv
from datetime import datetime
from typing import Dict, List, Tuple
import platform

import cv2
import numpy as np
import face_recognition
import pandas as pd

ENCODINGS_PATH = "encodings.pkl"
ATTENDANCE_CSV = "attendance.csv"


def load_encodings(path: str) -> Dict[str, List]:
    if not os.path.exists(path):
        print("[WARN] encodings.pkl not found. Please run register_student.py first.")
        return {"encodings": [], "names": []}
    try:
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or "encodings" not in data or "names" not in data:
            print("[WARN] encodings.pkl has invalid format. Recreate using register_student.py.")
            return {"encodings": [], "names": []}
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load encodings: {e}")
        return {"encodings": [], "names": []}


def ensure_attendance_csv(path: str) -> None:
    if not os.path.exists(path):
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])  # headers
        print(f"[INFO] Created {path} with headers.")


def has_marked_today(name: str, path: str) -> bool:
    today = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
        if df.empty:
            return False
        df_today = df[(df["Name"] == name) & (df["Date"] == today)]
        return not df_today.empty
    except Exception:
        try:
            with open(path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Name") == name and row.get("Date") == today:
                        return True
        except Exception:
            return False
    return False


def mark_attendance(name: str, path: str) -> None:
    ensure_attendance_csv(path)
    if has_marked_today(name, path):
        return
    now = datetime.now()
    row = [name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")]
    try:
        with open(path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"[INFO] Marked attendance: {row}")
    except Exception as e:
        print(f"[ERROR] Failed to write attendance: {e}")


def _ensure_c_uint8(img: np.ndarray) -> np.ndarray:
    """Force numpy array to be uint8, C-contiguous, owned memory (dlib compatibility)."""
    return np.array(img, dtype=np.uint8, order='C')


def _detect_faces_fallback(rgb_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Fallback detector using OpenCV Haar cascades. Returns (top,right,bottom,left)."""
    try:
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    except Exception:
        return []
    cascade_path = getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        return []
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    boxes: List[Tuple[int, int, int, int]] = []
    for (x, y, w, h) in rects:
        boxes.append((y, x + w, y + h, x))
    return boxes


def _backend_code(name: str) -> int:
    name = (name or "").lower()
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    if name == "any":
        return cv2.CAP_ANY
    # auto: prefer dshow on Windows, else ANY
    if platform.system().lower().startswith("win"):
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def _open_camera(index: int, backend: str) -> cv2.VideoCapture:
    prefs = []
    first = _backend_code(backend)
    prefs.append(first)
    # add fallbacks
    if first != cv2.CAP_MSMF:
        prefs.append(cv2.CAP_MSMF)
    if first != cv2.CAP_ANY:
        prefs.append(cv2.CAP_ANY)

    cap = None
    for api in prefs:
        try:
            c = cv2.VideoCapture(index, api)
            if c.isOpened():
                print(f"[INFO] Opened camera {index} with backend={api}")
                # Hint a reasonable resolution
                c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return c
            c.release()
        except Exception:
            pass
    # last resort
    c = cv2.VideoCapture(index)
    if c.isOpened():
        print(f"[INFO] Opened camera {index} with default backend")
        return c
    return c

# ---- Robust matching helpers ----

def _compute_centroids(known_encodings: List[np.ndarray], known_names: List[str]) -> Dict[str, np.ndarray]:
    by_name: Dict[str, List[np.ndarray]] = {}
    for enc, nm in zip(known_encodings, known_names):
        by_name.setdefault(nm, []).append(np.asarray(enc, dtype=np.float32))
    cents: Dict[str, np.ndarray] = {}
    for nm, vecs in by_name.items():
        if vecs:
            cents[nm] = np.mean(np.stack(vecs, axis=0), axis=0)
    return cents


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def _robust_match(enc: np.ndarray, known_encodings: List[np.ndarray], known_names: List[str], tolerance: float) -> Tuple[str, float]:
    if not known_encodings:
        return "Unknown", 1.0
    dists = face_recognition.face_distance(known_encodings, enc)
    k = min(5, len(known_encodings))
    order = np.argsort(dists)
    top_idx = order[:k]
    votes: Dict[str, List[float]] = {}
    for i in top_idx:
        nm = known_names[i]
        votes.setdefault(nm, []).append(float(dists[i]))
    # winner by most votes, tiebreak by lower mean
    winner = None
    winner_mean = 1.0
    winner_min = 1.0
    for nm, ds in votes.items():
        m = float(np.mean(ds))
        md = float(np.min(ds))
        if (winner is None) or (len(ds) > len(votes.get(winner, []))) or (len(ds) == len(votes.get(winner, [])) and m < winner_mean):
            winner = nm; winner_mean = m; winner_min = md
    others = [(nm, float(np.mean(ds))) for nm, ds in votes.items() if nm != winner]
    others.sort(key=lambda x: x[1])
    second_mean = others[0][1] if others else 1.0

    # centroid check
    centroids = _compute_centroids(known_encodings, known_names)
    cent = centroids.get(winner)
    cent_dist = _euclid(enc, cent) if cent is not None else 1.0

    margin = 0.08
    ok_nn = winner_min <= 0.52
    ok_mean = winner_mean <= tolerance
    ok_margin = (second_mean - winner_mean) >= margin
    ok_centroid = cent_dist <= (tolerance + 0.02)
    votes_needed = 3 if k >= 5 else max(1, int(np.ceil(k * 0.6)))
    ok_votes = len(votes.get(winner, [])) >= votes_needed

    if ok_nn and ok_mean and ok_margin and ok_centroid and ok_votes:
        return winner, winner_mean
    return "Unknown", 1.0

# ---------------------------------

def recognize_and_log(camera_index: int, tolerance: float, headless: bool, backend: str) -> None:
    data = load_encodings(ENCODINGS_PATH)
    known_encodings = data["encodings"]
    known_names = data["names"]

    if len(known_encodings) == 0:
        print("[ERROR] No known encodings loaded. Register students first.")
        return

    ensure_attendance_csv(ATTENDANCE_CSV)

    # Reduce OpenCV log noise
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass

    cap = _open_camera(camera_index, backend)
    if not cap or not cap.isOpened():
        print("[ERROR] Cannot open webcam. Try --camera-index 1 or --backend dshow/msmf.")
        return

    print("[INFO] Press 'q' to quit.")

    verify_name = None
    verify_count = 0
    marked = False

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Resize and convert
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        rgb_small = _ensure_c_uint8(rgb_small)

        # Detect faces
        try:
            boxes = face_recognition.face_locations(rgb_small, model="hog")
        except Exception:
            boxes = []
        if not boxes:
            boxes = _detect_faces_fallback(rgb_small)

        # Choose the largest face only
        if boxes:
            def area(b):
                t, r, btm, l = b
                return abs(btm - t) * abs(r - l)
            boxes = [sorted(boxes, key=area, reverse=True)[0]]

        try:
            encodings = face_recognition.face_encodings(rgb_small, boxes)
        except Exception:
            encodings = []

        name = "Unknown"
        if encodings:
            name, score = _robust_match(encodings[0], known_encodings, known_names, tolerance)

        # Multi-frame consensus: require 3 consecutive frames of same name
        if name != "Unknown":
            if verify_name == name:
                verify_count += 1
            else:
                verify_name = name
                verify_count = 1
        else:
            verify_name = None
            verify_count = 0

        if not headless:
            # draw box and status
            if boxes:
                t, r, btm, l = boxes[0]
                t *= 2; r *= 2; btm *= 2; l *= 2
                color = (0, 255, 0) if verify_name else (0, 0, 255)
                cv2.rectangle(frame, (l, t), (r, btm), color, 2)
                label = f"Verifying {verify_name} ({verify_count}/3)" if verify_name else "Unknown"
                cv2.rectangle(frame, (l, btm - 25), (r, btm), color, cv2.FILLED)
                cv2.putText(frame, label, (l + 6, btm - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Smart Attendance - Press 'q' to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Quit requested.")
                break

        if not marked and verify_name and verify_count >= 3:
            mark_attendance(verify_name, ATTENDANCE_CSV)
            marked = True
            print(f"[INFO] Verified and marked: {verify_name}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Recognize faces and log attendance in real-time.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--tolerance", type=float, default=0.6, help="Face match tolerance (lower = stricter, default: 0.6)")
    parser.add_argument("--headless", action="store_true", help="Do not display video window")
    parser.add_argument("--backend", choices=["auto","dshow","msmf","any"], default="auto", help="Video backend to use (Windows: try dshow if ms-mf fails)")
    args = parser.parse_args()

    recognize_and_log(args.camera_index, args.tolerance, args.headless, args.backend)


if __name__ == "__main__":
    main()
