from fastapi import APIRouter, HTTPException
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import uuid
from storage.session_history import SessionHistory
from analytics.workout_analytics import WorkoutSession, WorkoutStatistics
from analytics.performance_analytics import PerformanceAnalytics
from analytics.fatigue_analytics import FatigueAnalytics
from analytics.progress_prediction import ProgressPredictor
from analytics.report_generator import ReportGenerator

router = APIRouter()
history = SessionHistory()

# ── Pydantic Models ──────────────────────────────────────
class SetModel(BaseModel):
    reps: int
    weight: float
    duration_sec: Optional[float] = None

class SessionCreate(BaseModel):
    exercise: str
    sets: list[SetModel]
    total_duration_min: float
    heart_rate_avg: Optional[float] = None
    calories_burned: Optional[float] = None
    notes: Optional[str] = ""
    date: Optional[datetime] = None

# ── Session Endpoints ────────────────────────────────────
@router.post("/sessions", status_code=201)
def add_session(data: SessionCreate):
    session = WorkoutSession(
        session_id=str(uuid.uuid4()),
        date=data.date or datetime.now(),
        exercise=data.exercise,
        sets=[s.dict() for s in data.sets],
        total_duration_min=data.total_duration_min,
        heart_rate_avg=data.heart_rate_avg,
        calories_burned=data.calories_burned,
        notes=data.notes or "",
    )
    history.add(session)
    return {"message": "Session saved", "session_id": session.session_id}

@router.get("/sessions")
def get_sessions(exercise: Optional[str] = None, last_n: Optional[int] = None):
    if exercise:
        sessions = history.by_exercise(exercise)
    elif last_n:
        sessions = history.last_n(last_n)
    else:
        sessions = history.all()
    return [
        {
            "session_id": s.session_id,
            "date": s.date.isoformat(),
            "exercise": s.exercise,
            "total_duration_min": s.total_duration_min,
            "calories_burned": s.calories_burned,
            "sets_count": len(s.sets),
        }
        for s in sessions
    ]

@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    history.delete(session_id)
    return {"message": "Deleted", "session_id": session_id}

# ── Analytics Endpoints ──────────────────────────────────
@router.get("/analytics/summary")
def get_summary():
    sessions = history.all()
    if not sessions:
        raise HTTPException(404, "No sessions found")
    stats = WorkoutStatistics(sessions)
    perf = PerformanceAnalytics(sessions)
    return {
        **stats.summary(),
        "consistency_score": perf.consistency_score(),
        "personal_records": perf.personal_records(),
    }

@router.get("/analytics/strength/{exercise}")
def strength_trend(exercise: str):
    sessions = history.by_exercise(exercise)
    if not sessions:
        raise HTTPException(404, f"No sessions for exercise: {exercise}")
    perf = PerformanceAnalytics(sessions)
    return perf.strength_trend(exercise)

@router.get("/analytics/fatigue/{exercise}")
def fatigue(exercise: str):
    sessions = history.all()
    fa = FatigueAnalytics(sessions)
    return fa.acute_chronic_workload_ratio(exercise)

@router.get("/analytics/predict/{exercise}")
def predict(exercise: str, future_sessions: int = 10):
    sessions = history.by_exercise(exercise)
    if len(sessions) < 3:
        raise HTTPException(400, "Need at least 3 sessions for prediction")
    predictor = ProgressPredictor(sessions)
    return predictor.predict_max_weight(exercise, future_sessions)

@router.get("/analytics/plateau/{exercise}")
def plateau(exercise: str):
    sessions = history.by_exercise(exercise)
    predictor = ProgressPredictor(sessions)
    return predictor.plateau_detection(exercise)

# ── Report Endpoint ──────────────────────────────────────
@router.get("/report")
def get_report(format: str = "json"):
    sessions = history.all()
    if not sessions:
        raise HTTPException(404, "No sessions found")
    rg = ReportGenerator(sessions)
    if format == "text":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(rg.to_text())
    return rg.generate()