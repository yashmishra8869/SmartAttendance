#!/usr/bin/env python3
"""
AI-based Smart Attendance System - Student Registration

Usage examples:
1) Capture from webcam (default camera index 0):
   python register_student.py --name "Alice Johnson" --from-webcam --samples 15

2) Register from a folder of existing images (JPG/PNG, 8-bit):
   python register_student.py --name "Bob Smith" --images-dir ./images/bob

3) Register from specific image files (wildcards supported):
   python register_student.py --name "Charlie" --images img1.jpg img2.png "./mix/*.PNG"

4) If a student already exists and you want to append more samples:
   python register_student.py --name "Alice Johnson" --append --images-dir ./more_alice

5) If a student already exists and you want to replace their data entirely:
   python register_student.py --name "Alice Johnson" --replace --from-webcam --samples 20

Dependencies:
- Install required libraries first:
    pip install -r requirements.txt

Notes:
- This script extracts face embeddings using the face_recognition library.
- It stores all embeddings and names in encodings.pkl at the project root.
- Images should be normal 8-bit JPG/PNG. Frames without a detectable single face are skipped.
- Handle duplicate names with --append or --replace. Without these, duplicates are blocked.
"""

import argparse
import os
import sys
import pickle
import glob
from typing import List, Tuple, Optional
from pathlib import Path

# Graceful imports with helpful messages
try:
    import cv2
except ImportError:
    print("[ERROR] Missing dependency: opencv-python. Install with: pip install -r requirements.txt")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("[ERROR] Missing dependency: numpy. Install with: pip install -r requirements.txt")
    sys.exit(1)

try:
    import face_recognition
except ImportError:
    print("[ERROR] Missing dependency: face_recognition. Install with: pip install -r requirements.txt")
    sys.exit(1)

ENCODINGS_PATH = "encodings.pkl"
ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEBUG = False


def load_encodings(path: str) -> dict:
    """Load encodings pickle if it exists, otherwise return empty structure."""
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            # Basic structure validation
            if not isinstance(data, dict) or "encodings" not in data or "names" not in data:
                print("[WARN] encodings.pkl has unexpected format. Reinitializing.")
                return {"encodings": [], "names": []}
            return data
        except Exception as e:
            print(f"[ERROR] Failed to load encodings: {e}")
            return {"encodings": [], "names": []}
    else:
        return {"encodings": [], "names": []}


def save_encodings(path: str, data: dict) -> None:
    """Save encodings structure to pickle."""
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] Saved encodings to {path}")
    except Exception as e:
        print(f"[ERROR] Failed to save encodings: {e}")


def _collect_from_dir(dir_path: Path) -> List[str]:
    results: List[str] = []
    # Recursive search; filter by allowed extensions (case-insensitive)
    for p in dir_path.rglob('*'):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            results.append(str(p))
    return results


def get_image_paths(images: Optional[List[str]], images_dir: Optional[str]) -> List[str]:
    """Collect image paths from explicit list (supports wildcards and directories) or a directory (recursive)."""
    paths: List[str] = []

    # From explicit items
    if images:
        for item in images:
            if not item:
                continue
            # Expand wildcards if present
            if any(ch in item for ch in ['*', '?', '[']):
                for m in glob.glob(item):
                    mp = Path(m)
                    if mp.is_file() and mp.suffix.lower() in ALLOWED_EXTS:
                        paths.append(str(mp))
                    elif mp.is_dir():
                        paths.extend(_collect_from_dir(mp))
                continue

            p = Path(item)
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                paths.append(str(p))
            elif p.is_dir():
                paths.extend(_collect_from_dir(p))
            else:
                print(f"[WARN] Not found or unsupported type, skipping: {item}")

    # From directory param
    if images_dir:
        d = Path(images_dir)
        if d.is_dir():
            paths.extend(_collect_from_dir(d))
        else:
            print(f"[WARN] Not a directory or not found: {images_dir}")

    # Deduplicate while preserving order
    seen = set()
    unique_paths = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    return unique_paths


def to_rgb_uint8(image: np.ndarray) -> Optional[np.ndarray]:
    """Convert any OpenCV-loaded image to RGB uint8 (handle grayscale, BGRA, 16-bit) and ensure C-contiguous."""
    if image is None or image.size == 0:
        return None

    # Convert bit depth to 8-bit if needed
    if image.dtype != np.uint8:
        try:
            if image.dtype == np.uint16:
                image = cv2.convertScaleAbs(image, alpha=255.0 / 65535.0)
            else:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] Failed to normalize bit depth: {e}")
            return None

    try:
        if len(image.shape) == 2:  # Grayscale
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
        # Ensure uint8 and C-contiguous
        if rgb.dtype != np.uint8:
            rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return np.ascontiguousarray(rgb)
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] Failed colorspace conversion: {e}")
        return None


def detect_faces_opencv_haar(img_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Fallback face detector using OpenCV Haar cascades. Returns list of (top,right,bottom,left)."""
    try:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    except Exception:
        return []
    cascade_path = getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_default.xml'
    if not cascade_path or not os.path.exists(cascade_path):
        if DEBUG:
            print(f"[DEBUG] Haar cascade not found at: {cascade_path}")
        return []
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        if DEBUG:
            print("[DEBUG] Failed to load Haar cascade classifier")
        return []
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    boxes: List[Tuple[int, int, int, int]] = []
    for (x, y, w, h) in rects:
        top, right, bottom, left = y, x + w, y + h, x
        boxes.append((top, right, bottom, left))
    if DEBUG and boxes:
        print(f"[DEBUG] Haar detected faces: {len(boxes)}")
    return boxes


def detect_faces_multi_attempt(img_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Try multiple strategies to detect faces: HOG with upsample, then CNN, optional scale-up, then Haar fallback."""
    if img_rgb is None:
        return []
    # Ensure C-contiguous, own data, uint8 for dlib compatibility
    img_rgb = np.array(img_rgb, dtype=np.uint8, order='C')

    h, w = img_rgb.shape[:2]
    attempts = []
    # HOG upsample attempts
    attempts.append(("hog", 1, img_rgb))
    attempts.append(("hog", 2, img_rgb))

    # If small image, scale up and try HOG again
    if max(h, w) < 400:
        scale = 2
        big = cv2.resize(img_rgb, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        big = np.array(big, dtype=np.uint8, order='C')
        attempts.append(("hog", 1, big))

    # CNN attempts (can be slower but more robust)
    attempts.append(("cnn", 0, img_rgb))

    for model, up, img in attempts:
        try:
            # Ensure each attempt image meets dlib expectations
            img_try = np.array(img, dtype=np.uint8, order='C')
            boxes = face_recognition.face_locations(img_try, number_of_times_to_upsample=up, model=model)
            if boxes:
                # If we resized image, scale boxes back down
                if img_try.shape[:2] != img_rgb.shape[:2]:
                    sy = img_rgb.shape[0] / img_try.shape[0]
                    sx = img_rgb.shape[1] / img_try.shape[1]
                    scaled = []
                    for (t, r, b, l) in boxes:
                        scaled.append((int(t * sy), int(r * sx), int(b * sy), int(l * sx)))
                    boxes = scaled
                if DEBUG:
                    print(f"[DEBUG] Detected faces with model={model}, upsample={up}, count={len(boxes)}")
                return boxes
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] face_locations failed (model={model}, up={up}): {e}")

    # OpenCV Haar fallback
    haar_boxes = detect_faces_opencv_haar(img_rgb)
    if haar_boxes:
        return haar_boxes

    return []


def extract_face_encoding_from_image(img_any: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Return one face encoding and its location from the image, preferring the largest face.
    Accepts images with various bit depths and channel counts; converts to RGB uint8.
    Also tries rotated variants (90/180/270 deg) if no face found.
    Returns (encoding, location). If not found, returns (None, None).
    """
    img_rgb = to_rgb_uint8(img_any)
    if img_rgb is None:
        if DEBUG:
            print("[DEBUG] to_rgb_uint8 returned None")
        return None, None

    # Candidates: original + rotated versions
    candidates: List[Tuple[np.ndarray, str]] = [(img_rgb, "orig")]
    try:
        rot90 = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
        rot270 = cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rot180 = cv2.rotate(img_rgb, cv2.ROTATE_180)
        candidates.extend([(rot90, "rot90"), (rot270, "rot270"), (rot180, "rot180")])
    except Exception:
        pass

    for cand_img, tag in candidates:
        if cand_img is None:
            continue
        # Ensure uint8, C-contiguous, own data
        cand_img = np.array(cand_img, dtype=np.uint8, order='C')

        boxes = detect_faces_multi_attempt(cand_img)
        if not boxes:
            continue

        # Choose largest face by area
        def area(box):
            top, right, bottom, left = box
            return abs(bottom - top) * abs(right - left)

        box = sorted(boxes, key=area, reverse=True)[0]

        try:
            img_for_enc = np.array(cand_img, dtype=np.uint8, order='C')
            encs = face_recognition.face_encodings(img_for_enc, [box])
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] face_encodings failed on {tag}: {type(e).__name__}: {e}")
            continue

        if encs:
            if DEBUG and tag != "orig":
                print(f"[DEBUG] Face encoded from {tag} orientation")
            return encs[0], box

    # If all attempts failed
    return None, None


def capture_encodings_from_webcam(samples: int, camera_index: int = 0) -> List[np.ndarray]:
    """Capture encodings from webcam. Attempts to collect `samples` encodings where exactly one face is visible."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check camera index or permissions.")
        return []

    collected: List[np.ndarray] = []
    print("[INFO] Press 'q' to quit early.")

    while len(collected) < samples:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from camera.")
            break

        full_rgb = to_rgb_uint8(frame)
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        small_rgb = to_rgb_uint8(small)

        message = "Show exactly ONE face. Tips: center face, good lighting, look at camera"
        color = (0, 0, 255)

        boxes_small = detect_faces_multi_attempt(small_rgb) if small_rgb is not None else []
        src = "small"
        boxes_full: List[Tuple[int, int, int, int]] = []

        if not boxes_small and full_rgb is not None:
            boxes_full = detect_faces_multi_attempt(full_rgb)
            src = "full"

        # Prefer whichever found boxes
        boxes = boxes_small if boxes_small else boxes_full

        # Draw guidance rectangles
        draw_box = None
        if boxes:
            if src == "small":
                t, r, b, l = boxes[0]
                top, right, bottom, left = t * 2, r * 2, b * 2, l * 2
            else:
                top, right, bottom, left = boxes[0]
            draw_box = (left, top, right, bottom)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

        # Logic for capturing exactly one face
        faces_count = len(boxes)
        if faces_count == 1 and full_rgb is not None:
            if src == "small":
                t, r, b, l = boxes[0]
                top, right, bottom, left = t * 2, r * 2, b * 2, l * 2
            else:
                top, right, bottom, left = boxes[0]
            # Optional: reject tiny faces (<80px height)
            if (bottom - top) < 80:
                message = "Move closer to the camera"
            else:
                enc = face_recognition.face_encodings(full_rgb, [(top, right, bottom, left)])
                if enc:
                    collected.append(enc[0])
                    message = f"Captured sample {len(collected)}/{samples}"
                    color = (0, 255, 0)
                else:
                    message = "Face detected but encoding failed"
        elif faces_count > 1:
            message = "Multiple faces detected - only one person should be in view"
        else:
            message = "No clear face - improve lighting and center your face"

        cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Registration - Press 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Early quit requested.")
            break

    cap.release()
    cv2.destroyAllWindows()

    return collected


def encodings_from_image_files(paths: List[str]) -> List[np.ndarray]:
    """Load images using face_recognition (PIL backend) to ensure RGB uint8 layout, then detect and encode.
    Also tries 90/180/270 degree rotations if needed.
    """
    encs: List[np.ndarray] = []
    for p in paths:
        try:
            img = face_recognition.load_image_file(p)  # RGB, uint8
            if img is None:
                print(f"[WARN] Failed to read image: {p}")
                continue
            img = np.array(img, dtype=np.uint8, order='C')
            if DEBUG:
                print(f"[DEBUG] Loaded (fr) {p}: dtype={img.dtype}, shape={getattr(img,'shape',None)}")
        except Exception as e:
            print(f"[WARN] Failed to read image with face_recognition: {p} ({e})")
            continue

        candidates: List[Tuple[np.ndarray, str]] = [(img, "orig")]
        try:
            candidates.extend([
                (np.ascontiguousarray(np.rot90(img, 1)), "rot90"),
                (np.ascontiguousarray(np.rot90(img, 2)), "rot180"),
                (np.ascontiguousarray(np.rot90(img, 3)), "rot270"),
            ])
        except Exception:
            pass

        encoded = False
        for cand, tag in candidates:
            boxes = detect_faces_multi_attempt(cand)
            if not boxes:
                continue

            # Pick largest face
            def area(b):
                t, r, btm, l = b
                return abs(btm - t) * abs(r - l)

            box = sorted(boxes, key=area, reverse=True)[0]
            try:
                enc_list = face_recognition.face_encodings(cand, [box])
            except Exception as e:
                if DEBUG:
                    print(f"[DEBUG] face_encodings failed ({tag}) on {p}: {type(e).__name__}: {e}")
                continue
            if enc_list:
                encs.append(enc_list[0])
                print(f"[INFO] Processed: {p} ({tag})")
                encoded = True
                break

        if not encoded:
            print(f"[WARN] No usable single face in: {p} (ensure face is clear, frontal, >=100x100, good lighting)")

    return encs


def main():
    parser = argparse.ArgumentParser(description="Register a student's face encodings for attendance.")
    parser.add_argument("--name", required=True, help="Full name of the student to register")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-webcam", action="store_true", help="Capture samples from webcam")
    mode.add_argument("--images-dir", type=str, help="Directory containing face images (recursive)")
    mode.add_argument("--images", nargs="*", help="Specific image file(s) or patterns (supports wildcards)")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to capture from webcam (default: 10)")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs for image format issues")
    dup = parser.add_mutually_exclusive_group()
    dup.add_argument("--append", action="store_true", help="Append new samples if the name already exists")
    dup.add_argument("--replace", action="store_true", help="Replace existing samples for this name")

    args = parser.parse_args()

    global DEBUG
    DEBUG = bool(args.debug)

    name = args.name.strip()
    if not name:
        print("[ERROR] Name cannot be empty.")
        sys.exit(1)

    data = load_encodings(ENCODINGS_PATH)
    existing_names = data["names"]
    has_duplicate = name in existing_names

    if has_duplicate and not (args.append or args.replace):
        print("[ERROR] This name already exists. Use --append to add more samples or --replace to overwrite.")
        sys.exit(1)

    # Collect encodings
    new_encs: List[np.ndarray] = []
    if args.from_webcam:
        new_encs = capture_encodings_from_webcam(samples=max(1, args.samples), camera_index=args.camera_index)
    else:
        paths = get_image_paths(args.images, args.images_dir)
        if not paths:
            cwd = os.getcwd()
            print("[ERROR] No images found to process.")
            if args.images_dir:
                print(f"[HINT] Checked directory (recursive): {Path(args.images_dir).resolve()} | CWD: {cwd}")
                print("[HINT] Allowed extensions: .jpg, .jpeg, .png, .bmp, .webp")
                print("[HINT] Example: python register_student.py --name \"Alice\" --images-dir ./images/alice")
            if args.images is not None:
                print("[HINT] You can pass wildcards, directories, or files to --images, e.g. --images ./folder \"./mix/*.PNG\" img1.jpg")
            sys.exit(1)
        new_encs = encodings_from_image_files(paths)

    if not new_encs:
        print("[ERROR] No encodings were captured. Nothing to save.")
        print("[HINT] Try: --debug to see image dtype/shape, use a clear frontal face, bigger image (>= 400px on longer side), or --from-webcam")
        sys.exit(1)

    # Apply duplicate strategy
    if has_duplicate and args.replace:
        kept_encs = []
        kept_names = []
        removed = 0
        for enc, nm in zip(data["encodings"], data["names"]):
            if nm != name:
                kept_encs.append(enc)
                kept_names.append(nm)
            else:
                removed += 1
        data["encodings"] = kept_encs
        data["names"] = kept_names
        print(f"[INFO] Replacing existing samples for '{name}'. Removed {removed} old samples.")

    # Append new samples
    data["encodings"].extend(new_encs)
    data["names"].extend([name] * len(new_encs))

    save_encodings(ENCODINGS_PATH, data)

    print(f"[SUCCESS] Registered '{name}' with {len(new_encs)} new sample(s).")
    print(f"[INFO] Total samples in database: {len(data['encodings'])}")


if __name__ == "__main__":
    main()
