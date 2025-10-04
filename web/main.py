from pathlib import Path
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, Request, Body
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
import pandas as pd

from .face_utils import (
    ROOT_DIR,
    ENCODINGS_PATH,
    ATTENDANCE_CSV,
    load_encodings,
    save_encodings,
    extract_face_encoding_from_image,
    recognize_from_image,
    ensure_attendance_csv,
    mark_attendance,
    compute_monthly_attendance,
    imdecode_image,
    decode_base64_image_uri,
    recognize_from_image_with_debug,
    remove_students_by_names,
    has_marked_today,
    get_today_mark_time,
    compute_monthly_presence,
    normalize_name,
)

app = FastAPI(title="SmartAttendanceWeb")

static_dir = Path(__file__).resolve().parent / "static"
templates_dir = Path(__file__).resolve().parent / "templates"
static_dir.mkdir(parents=True, exist_ok=True)
templates_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))


@app.get("/health")
async def health():
    return {"status": "ok"}


def _students_with_counts() -> List[dict]:
    data = load_encodings(ENCODINGS_PATH)
    names = data.get("names", [])
    # count samples per normalized name
    counts = {}
    for nm in names:
        nn = normalize_name(nm)
        counts[nn] = counts.get(nn, 0) + 1
    return [{"name": nm, "samples": cnt} for nm, cnt in sorted(counts.items(), key=lambda x: x[0].lower())]


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, month: str = None):
    ensure_attendance_csv(ATTENDANCE_CSV)
    # default to current month
    if not month:
        now = pd.Timestamp.now()
        month = f"{now.year:04d}-{now.month:02d}"
    try:
        year, mo = map(int, month.split("-"))
    except Exception:
        now = pd.Timestamp.now()
        year, mo = now.year, now.month
    stats = compute_monthly_attendance(ATTENDANCE_CSV, year, mo)
    matrix = compute_monthly_presence(ATTENDANCE_CSV, year, mo)
    return templates.TemplateResponse("dashboard.html", {"request": request, "month": f"{year:04d}-{mo:02d}", "stats": stats, "matrix": matrix})


@app.get("/students", response_class=HTMLResponse)
async def students(request: Request):
    return templates.TemplateResponse("students.html", {"request": request, "students": _students_with_counts()})


@app.post("/students/remove")
async def students_remove(names: str = Form(...)):
    # Accept comma-separated names in a single field
    raw = [p.strip() for p in names.split(",")]
    to_remove = [p for p in raw if p]
    if to_remove:
        remove_students_by_names(to_remove, ENCODINGS_PATH)
    # redirect back to students page
    return RedirectResponse(url="/students", status_code=303)


@app.get("/scan", response_class=HTMLResponse)
async def scan(request: Request):
    return templates.TemplateResponse("scan.html", {"request": request})


@app.get("/download/attendance.csv")
async def download_attendance():
    ensure_attendance_csv(ATTENDANCE_CSV)
    return FileResponse(path=str(ATTENDANCE_CSV), media_type="text/csv", filename="attendance.csv")


@app.get("/api/attendance")
async def api_attendance(month: str):
    try:
        year, mo = map(int, month.split("-"))
    except Exception:
        return JSONResponse(status_code=400, content={"error": "month must be YYYY-MM"})
    stats = compute_monthly_attendance(ATTENDANCE_CSV, year, mo)
    return stats


@app.post("/api/register")
async def api_register(name: str = Form(...), images: List[UploadFile] = File(...)):
    name = normalize_name(name)
    if not name:
        return JSONResponse(status_code=400, content={"error": "Name is required"})
    data = load_encodings(ENCODINGS_PATH)
    added = 0
    for up in images:
        content = await up.read()
        img = imdecode_image(content)
        if img is None:
            continue
        enc, _ = extract_face_encoding_from_image(img)
        if enc is None:
            continue
        data["encodings"].append(enc)
        data["names"].append(name)
        added += 1
    if added == 0:
        return JSONResponse(status_code=400, content={"error": "No usable faces found in uploads"})
    save_encodings(data, ENCODINGS_PATH)
    total = sum(1 for n in data["names"] if normalize_name(n) == name)
    return {"name": name, "added_samples": added, "total_samples_for_name": total}


@app.post("/api/register-webcam")
async def api_register_webcam(payload: dict = Body(...)):
    name = normalize_name(str(payload.get("name", "")))
    frames = payload.get("frames", [])  # array of base64 data URLs or plain base64
    if not name:
        return JSONResponse(status_code=400, content={"error": "Name is required"})
    if not isinstance(frames, list) or not frames:
        return JSONResponse(status_code=400, content={"error": "frames must be a non-empty array"})
    data = load_encodings(ENCODINGS_PATH)
    added = 0
    for b64 in frames:
        img = decode_base64_image_uri(b64)
        if img is None:
            continue
        enc, _ = extract_face_encoding_from_image(img)
        if enc is None:
            continue
        data["encodings"].append(enc)
        data["names"].append(name)
        added += 1
    if added == 0:
        return JSONResponse(status_code=400, content={"error": "No usable faces found in frames"})
    save_encodings(data, ENCODINGS_PATH)
    total = sum(1 for n in data["names"] if normalize_name(n) == name)
    return {"name": name, "added_samples": added, "total_samples_for_name": total}


@app.post("/api/scan")
async def api_scan(request: Request, file: UploadFile | None = File(None), payload: dict | None = Body(None)):
    # Accept either multipart file or JSON { image }
    debug = False
    data_json = None

    # Try JSON payload param first
    if payload and isinstance(payload, dict):
        data_json = payload
        debug = bool(payload.get("debug", False))

    # If still no JSON and content-type is JSON, read request body manually
    if data_json is None:
        try:
            ct = request.headers.get("content-type", "").lower()
            if ct.startswith("application/json"):
                data_json = await request.json()
                if isinstance(data_json, dict):
                    debug = bool(data_json.get("debug", False))
        except Exception:
            data_json = None

    img = None
    if file is not None:
        content = await file.read()
        img = imdecode_image(content)
    if img is None and isinstance(data_json, dict) and data_json.get("image"):
        img = decode_base64_image_uri(str(data_json.get("image")))

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Provide image as multipart file or JSON {image}"})

    data = load_encodings(ENCODINGS_PATH)
    if debug:
        name, info = recognize_from_image_with_debug(img, data.get("encodings", []), data.get("names", []))
    else:
        name = recognize_from_image(img, data.get("encodings", []), data.get("names", []))
        info = None

    if name != "Unknown":
        # already marked handling
        if has_marked_today(name, ATTENDANCE_CSV):
            prior = get_today_mark_time(name, ATTENDANCE_CSV)
            # format outputs
            date_fmt = pd.Timestamp.now().strftime('%d/%m/%Y')
            time_fmt = None
            if isinstance(prior, str) and prior:
                # prior like HH:MM:SS -> HH/MM
                parts = str(prior).split(':')
                if len(parts) >= 2:
                    time_fmt = f"{parts[0]}/{parts[1]}"
            res = {"name": name, "already_marked": True, "date": date_fmt, "time": time_fmt}
            return res
        mark_attendance(name, ATTENDANCE_CSV)
        now = pd.Timestamp.now()
        date_fmt = now.strftime('%d/%m/%Y')
        time_fmt = now.strftime('%H/%M')
        res = {"name": name, "date": date_fmt, "time": time_fmt}
        if info is not None:
            res["debug"] = info
        return res
    res = {"name": "Unknown"}
    if info is not None:
        res["debug"] = info
    return res


if __name__ == "__main__":
    uvicorn.run("web.main:app", host="0.0.0.0", port=8000, reload=True)
