from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from typing import Iterable, List, Optional, Dict
import os

from sqlalchemy import (
    create_engine, Column, Integer, String, Date, Time, DateTime, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

DB_URL = os.getenv("SMARTATTENDANCE_DB_URL", "").strip()

Base = declarative_base()
_engine = None
_SessionLocal = None


def _get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        if not DB_URL:
            raise RuntimeError("SMARTATTENDANCE_DB_URL is not set")
        _engine = create_engine(DB_URL, future=True, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
    return _engine


def get_session() -> Session:
    if _SessionLocal is None:
        _get_engine()
    return _SessionLocal()


class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    time = Column(Time, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("name", "date", name="uq_attendance_name_date"),
    )


def ensure_tables() -> None:
    eng = _get_engine()
    Base.metadata.create_all(eng)


# Name normalization should mirror face_utils.normalize_name; duplicated minimally here
_def_norm = lambda s: " ".join(str(s).split()).title()


def has_marked_today(name: str) -> bool:
    today = date.today()
    with get_session() as sess:
        q = sess.query(Attendance).filter(Attendance.name == _def_norm(name), Attendance.date == today)
        return sess.query(q.exists()).scalar() or False


def get_today_mark_time(name: str) -> Optional[str]:
    today = date.today()
    with get_session() as sess:
        rec = (
            sess.query(Attendance)
            .filter(Attendance.name == _def_norm(name), Attendance.date == today)
            .first()
        )
        if not rec:
            return None
        try:
            return rec.time.strftime("%H:%M:%S")
        except Exception:
            return None


def mark_attendance(name: str) -> None:
    nm = _def_norm(name)
    today = date.today()
    now = datetime.now().time().replace(microsecond=0)
    with get_session() as sess:
        # check existing
        exists = (
            sess.query(Attendance)
            .filter(Attendance.name == nm, Attendance.date == today)
            .first()
        )
        if exists:
            return
        sess.add(Attendance(name=nm, date=today, time=now, created_at=datetime.utcnow()))
        try:
            sess.commit()
        except Exception:
            sess.rollback()
            # if race condition on unique constraint, ignore


def fetch_month(year: int, month: int) -> List[Dict]:
    from calendar import monthrange
    last = monthrange(year, month)[1]
    start = date(year, month, 1)
    end = date(year, month, last)
    with get_session() as sess:
        rows = (
            sess.query(Attendance)
            .filter(Attendance.date >= start, Attendance.date <= end)
            .order_by(Attendance.date.asc(), Attendance.time.asc(), Attendance.name.asc())
            .all()
        )
        return [
            {"Name": r.name, "Date": r.date, "Time": r.time.strftime("%H:%M:%S")}
            for r in rows
        ]


def fetch_all() -> List[Dict]:
    with get_session() as sess:
        rows = sess.query(Attendance).order_by(Attendance.date.asc(), Attendance.time.asc(), Attendance.name.asc()).all()
        return [
            {"Name": r.name, "Date": r.date, "Time": r.time.strftime("%H:%M:%S")}
            for r in rows
        ]


def recent_rows(limit: int = 5) -> List[Dict]:
    with get_session() as sess:
        rows = (
            sess.query(Attendance)
            .order_by(Attendance.date.desc(), Attendance.time.desc(), Attendance.id.desc())
            .limit(int(limit))
            .all()
        )
        # return newest first like tail
        rows = list(rows)
        return [
            {"Name": r.name, "Date": r.date, "Time": r.time.strftime("%H:%M:%S")}
            for r in rows
        ]
