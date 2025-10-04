import base64
import io
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import face_recognition
import math
import calendar

# Paths resolved relative to project root (one level up from this file)
ROOT_DIR = Path(__file__).resolve().parents[1]
ENCODINGS_PATH = ROOT_DIR / "encodings.pkl"
ATTENDANCE_CSV = ROOT_DIR / "attendance.csv"

# Locks to protect concurrent writes from the web server
enc_lock = threading.Lock()
att_lock = threading.Lock()

DEBUG = False
DEFAULT_TOLERANCE = 0.62


def load_encodings(path: Path = ENCODINGS_PATH) -> Dict[str, List]:
    import pickle
    if not path.exists():
        return {"encodings": [], "names": []}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or "encodings" not in data or "names" not in data:
            return {"encodings": [], "names": []}
        return data
    except Exception:
        return {"encodings": [], "names": []}


def save_encodings(data: Dict[str, List], path: Path = ENCODINGS_PATH) -> None:
    import pickle
    with enc_lock:
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def ensure_c_uint8(img: np.ndarray) -> np.ndarray:
    return np.array(img, dtype=np.uint8, order="C")


def _detect_faces_opencv_haar(img_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
    try:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    except Exception:
        return []
    cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        return []
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    boxes: List[Tuple[int, int, int, int]] = []
    for (x, y, w, h) in rects:
        top, right, bottom, left = y, x + w, y + h, x
        boxes.append((top, right, bottom, left))
    return boxes


def detect_faces_multi_attempt(img_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
    if img_rgb is None:
        return []
    img_rgb = ensure_c_uint8(img_rgb)
    h, w = img_rgb.shape[:2]
    attempts = [("hog", 1, img_rgb), ("hog", 2, img_rgb), ("hog", 3, img_rgb)]
    if max(h, w) < 400:
        scale = 2
        big = cv2.resize(img_rgb, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        attempts.append(("hog", 1, ensure_c_uint8(big)))
    attempts.append(("cnn", 0, img_rgb))

    for model, up, image in attempts:
        try:
            img_try = ensure_c_uint8(image)
            boxes = face_recognition.face_locations(img_try, number_of_times_to_upsample=up, model=model)
            if boxes:
                if img_try.shape[:2] != img_rgb.shape[:2]:
                    sy = img_rgb.shape[0] / img_try.shape[0]
                    sx = img_rgb.shape[1] / img_try.shape[1]
                    boxes = [(int(t * sy), int(r * sx), int(b * sy), int(l * sx)) for (t, r, b, l) in boxes]
                return boxes
        except Exception:
            continue

    return _detect_faces_opencv_haar(img_rgb)


def _to_rgb_uint8(image: np.ndarray) -> Optional[np.ndarray]:
    if image is None or image.size == 0:
        return None
    if image.dtype != np.uint8:
        try:
            if image.dtype == np.uint16:
                image = cv2.convertScaleAbs(image, alpha=255.0 / 65535.0)
            else:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        except Exception:
            return None
    try:
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            h, w, c = image.shape
            if c == 4:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif c == 3:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                return None
        else:
            return None
        return ensure_c_uint8(rgb)
    except Exception:
        return None

# ---- Robust matching helpers (same as CLI) ----

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


def _robust_match(enc: np.ndarray, known_encodings: List[np.ndarray], known_names: List[str], tolerance: float) -> Tuple[str, float, Dict]:
    if not known_encodings:
        return "Unknown", 1.0, {}
    dists = face_recognition.face_distance(known_encodings, enc)
    k = min(5, len(known_encodings))
    order = np.argsort(dists)
    top_idx = order[:k]
    votes: Dict[str, List[float]] = {}
    for i in top_idx:
        nm = known_names[i]
        votes.setdefault(nm, []).append(float(dists[i]))
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

    accepted = ok_nn and ok_mean and ok_margin and ok_centroid and ok_votes
    info = {
        "winner": winner,
        "winner_mean": winner_mean,
        "winner_min": winner_min,
        "second_mean": second_mean,
        "centroid_dist": cent_dist,
        "votes": {nm: {"count": len(ds), "mean": float(np.mean(ds))} for nm, ds in votes.items()},
        "criteria": {"ok_nn": ok_nn, "ok_mean": ok_mean, "ok_margin": ok_margin, "ok_centroid": ok_centroid, "ok_votes": ok_votes},
        "accepted": accepted,
    }

    if accepted:
        return winner, winner_mean, info
    return "Unknown", 1.0, info

# ----------------------------------------------

def extract_face_encoding_from_image(img_any: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    img_rgb = _to_rgb_uint8(img_any)
    if img_rgb is None:
        return None, None

    candidates: List[Tuple[np.ndarray, str]] = [(img_rgb, "orig")]
    try:
        candidates.extend([
            (cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE), "rot90"),
            (cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE), "rot270"),
            (cv2.rotate(img_rgb, cv2.ROTATE_180), "rot180"),
        ])
    except Exception:
        pass

    for cand_img, _ in candidates:
        cand_img = ensure_c_uint8(cand_img)
        boxes = detect_faces_multi_attempt(cand_img)
        if not boxes:
            continue
        # largest face
        box = sorted(boxes, key=lambda b: abs(b[2] - b[0]) * abs(b[1] - b[3]), reverse=True)[0]
        try:
            encs = face_recognition.face_encodings(cand_img, [box])
        except Exception:
            continue
        if encs:
            return encs[0], box
    return None, None


def recognize_from_image(img_any: np.ndarray, known_encodings: List[np.ndarray], known_names: List[str], tolerance: float = DEFAULT_TOLERANCE) -> str:
    # Follow the same robust pipeline as the CLI: downscale, HOG, largest face, robust match.
    if not isinstance(known_encodings, list) or not isinstance(known_names, list) or len(known_encodings) == 0 or len(known_encodings) != len(known_names):
        return "Unknown"

    img_rgb = _to_rgb_uint8(img_any)
    if img_rgb is None:
        return "Unknown"

    try:
        small = cv2.resize(img_rgb, (0, 0), fx=0.5, fy=0.5)
    except Exception:
        small = img_rgb
    small = ensure_c_uint8(small)

    try:
        boxes = face_recognition.face_locations(small, model="hog")
    except Exception:
        boxes = []
    if not boxes:
        # fallback
        boxes = detect_faces_multi_attempt(img_rgb)
        if boxes and small is not img_rgb:
            h, w = small.shape[:2]
            scaled = []
            for (t, r, b, l) in boxes:
                scaled.append((max(0, t//2), min(w, r//2), max(0, b//2), max(0, l//2)))
            boxes = scaled

    if not boxes:
        return "Unknown"

    # choose largest face only
    def area(b):
        t, r, btm, l = b
        return abs(btm - t) * abs(r - l)
    box = sorted(boxes, key=area, reverse=True)[0]

    try:
        encs = face_recognition.face_encodings(small, [box])
    except Exception:
        encs = []
    if not encs:
        return "Unknown"

    name, _score, _info = _robust_match(encs[0], known_encodings, known_names, tolerance)
    return name


def recognize_from_image_with_debug(img_any: np.ndarray, known_encodings: List[np.ndarray], known_names: List[str], tolerance: float = DEFAULT_TOLERANCE) -> Tuple[str, Dict]:
    info: Dict = {"detector": "hog", "boxes": 0, "encodings": 0}

    if not isinstance(known_encodings, list) or not isinstance(known_names, list) or len(known_encodings) == 0 or len(known_names) != len(known_encodings):
        info["error"] = "no_known_encodings"
        return "Unknown", info

    img_rgb = _to_rgb_uint8(img_any)
    if img_rgb is None:
        info["error"] = "invalid_image"
        return "Unknown", info

    try:
        small = cv2.resize(img_rgb, (0, 0), fx=0.5, fy=0.5)
    except Exception:
        small = img_rgb
    small = ensure_c_uint8(small)

    try:
        boxes = face_recognition.face_locations(small, model="hog")
    except Exception:
        boxes = []
    detector_used = "hog"
    if not boxes:
        detector_used = "fallback"
        boxes = detect_faces_multi_attempt(img_rgb)
        if boxes and small is not img_rgb:
            h, w = small.shape[:2]
            scaled = []
            for (t, r, b, l) in boxes:
                scaled.append((max(0, t//2), min(w, r//2), max(0, b//2), max(0, l//2)))
            boxes = scaled

    info["detector"] = detector_used
    info["boxes"] = len(boxes)

    if not boxes:
        return "Unknown", info

    # largest face only
    def area(b):
        t, r, btm, l = b
        return abs(btm - t) * abs(r - l)
    box = sorted(boxes, key=area, reverse=True)[0]

    try:
        encs = face_recognition.face_encodings(small, [box])
    except Exception:
        encs = []
    info["encodings"] = len(encs)

    if not encs:
        return "Unknown", info

    name, score, details = _robust_match(encs[0], known_encodings, known_names, tolerance)
    info.update(details)
    return name, info


def ensure_attendance_csv(path: Path = ATTENDANCE_CSV) -> None:
    if not path.exists():
        with att_lock:
            if not path.exists():
                path.write_text("Name,Date,Time\n", encoding="utf-8")


def _has_marked_today(name: str, path: Path = ATTENDANCE_CSV) -> bool:
    ensure_attendance_csv(path)
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty:
        return False
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    df_today = df[(df["Name"].astype(str) == str(name)) & (df["Date"] == today)]
    return not df_today.empty


# New: public helper to check today's mark
def has_marked_today(name: str, path: Path = ATTENDANCE_CSV) -> bool:
    return _has_marked_today(name, path)


# New: get the recorded time for today's mark if it exists
def get_today_mark_time(name: str, path: Path = ATTENDANCE_CSV) -> Optional[str]:
    ensure_attendance_csv(path)
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    df_today = df[(df["Name"].astype(str) == str(name)) & (df["Date"] == today)]
    if df_today.empty:
        return None
    # Return the earliest time of today for that name
    try:
        return str(df_today.iloc[0]["Time"])
    except Exception:
        return None


def mark_attendance(name: str, path: Path = ATTENDANCE_CSV) -> None:
    ensure_attendance_csv(path)
    if _has_marked_today(name, path):
        return
    now = pd.Timestamp.now()
    row = f"{name},{now.strftime('%Y-%m-%d')},{now.strftime('%H:%M:%S')}\n"
    with att_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(row)


def compute_monthly_attendance(path: Path, year: int, month: int) -> List[Dict]:
    ensure_attendance_csv(path)
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except Exception:
        return []
    if df.empty:
        return []
    # Normalize Date column
    if df["Date"].dtype != "datetime64[ns]":
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])  # drop unparsable
    df_month = df[(df["Date"].dt.year == year) & (df["Date"].dt.month == month)]
    if df_month.empty:
        return []
    total_days = df_month["Date"].dt.date.nunique()
    present = (
        df_month.assign(Day=df_month["Date"].dt.date)
        .groupby(["Name", "Day"]).size()
        .groupby("Name").size()
        .rename("days_present")
        .reset_index()
    )
    results: List[Dict] = []
    for _, row in present.iterrows():
        results.append({
            "name": row["Name"],
            "days_present": int(row["days_present"]),
            "total_days": int(total_days),
            "percent": (float(row["days_present"]) / float(total_days) * 100.0) if total_days else 0.0,
        })
    # Include names with 0 days present (registered but absent entire month)
    enc = load_encodings()
    all_names = sorted(set(enc.get("names", [])))
    already = {r["name"] for r in results}
    for nm in all_names:
        if nm not in already:
            results.append({
                "name": nm,
                "days_present": 0,
                "total_days": int(total_days),
                "percent": 0.0,
            })
    # Sort by name
    results.sort(key=lambda x: x["name"].lower())
    return results


def compute_monthly_presence(path: Path, year: int, month: int) -> Dict:
    """Return a day-wise presence matrix for the given month.
    Output schema:
      {
        "days_labels": ["01", "02", ...],          # day numbers as strings
        "days_iso":    ["YYYY-MM-01", ...],         # ISO dates for the month days
        "names":       ["Alice", "Bob", ...],       # sorted unique names (registered ? present)
        "presence":    { name: [True/False, ...] },   # len == number of days in month
        "total_days":  int
      }
    """
    ensure_attendance_csv(path)
    # Build full list of calendar days in the month
    last_day = calendar.monthrange(year, month)[1]
    idx_dates = [pd.Timestamp(year=year, month=month, day=d).date() for d in range(1, last_day + 1)]
    days_labels = [f"{d:02d}" for d in range(1, last_day + 1)]
    days_iso = [f"{year:04d}-{month:02d}-{d:02d}" for d in range(1, last_day + 1)]

    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except Exception:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])  # empty

    if not df.empty:
        # Normalize Date to date only
        if df["Date"].dtype != "datetime64[ns]":
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])  # drop unparsable
        df["Day"] = df["Date"].dt.date
        df_month = df[(df["Date"].dt.year == year) & (df["Date"].dt.month == month)]
    else:
        df_month = df

    # Name universe = registered names ? names in this month's records
    enc = load_encodings()
    reg_names = set(enc.get("names", []))
    present_names = set(df_month["Name"].astype(str).unique()) if not df_month.empty else set()
    all_names = sorted((reg_names | present_names), key=lambda x: str(x).lower())

    # Map name -> set of days present
    present_by_name: Dict[str, set] = {nm: set() for nm in all_names}
    if not df_month.empty:
        # Deduplicate multiple marks same day
        dedup = df_month.drop_duplicates(subset=["Name", "Day"])
        for _, row in dedup.iterrows():
            nm = str(row["Name"]) if pd.notna(row["Name"]) else ""
            day = row["Day"]
            if nm in present_by_name:
                present_by_name[nm].add(day)
            else:
                present_by_name[nm] = {day}

    presence_matrix: Dict[str, List[bool]] = {}
    for nm in all_names:
        s = present_by_name.get(nm, set())
        presence_matrix[nm] = [(d in s) for d in idx_dates]

    return {
        "days_labels": days_labels,
        "days_iso": days_iso,
        "names": all_names,
        "presence": presence_matrix,
        "total_days": len(idx_dates),
    }


def imdecode_image(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    return img


def decode_base64_image_uri(uri: str) -> Optional[np.ndarray]:
    # Accept data URLs like "data:image/png;base64,...."
    if "," in uri:
        uri = uri.split(",", 1)[1]
    try:
        b = base64.b64decode(uri, validate=False)
        return imdecode_image(b)
    except Exception:
        return None


def remove_students_by_names(names: List[str], path: Path = ENCODINGS_PATH) -> Dict[str, int]:
    """Remove all encodings for the given names (case-sensitive exact match).
    Returns a dict of { name: removed_count } for names that were found.
    """
    targets = [str(n).strip() for n in names if str(n).strip()]
    if not targets:
        return {}
    data = load_encodings(path)
    encs = data.get("encodings", [])
    nms = data.get("names", [])
    if not encs or not nms or len(encs) != len(nms):
        return {}

    kept_encs: List[np.ndarray] = []
    kept_names: List[str] = []
    removed: Dict[str, int] = {}

    target_set = set(targets)
    for enc, nm in zip(encs, nms):
        nm_str = str(nm)
        if nm_str in target_set:
            removed[nm_str] = removed.get(nm_str, 0) + 1
            continue
        kept_encs.append(enc)
        kept_names.append(nm)

    if len(kept_encs) != len(encs):
        with enc_lock:
            from pickle import dump, HIGHEST_PROTOCOL
            with open(path, "wb") as f:
                dump({"encodings": kept_encs, "names": kept_names}, f, protocol=HIGHEST_PROTOCOL)
    return removed
