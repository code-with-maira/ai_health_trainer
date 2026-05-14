import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class WorkoutSession:
    session_id: str
    date: datetime
    exercise: str
    sets: list[dict]           # [{"reps": 10, "weight": 60, "duration_sec": 45}]
    total_duration_min: float
    heart_rate_avg: Optional[float] = None
    calories_burned: Optional[float] = None
    notes: str = ""

class WorkoutStatistics:
    def __init__(self, sessions: list[WorkoutSession]):
        self.sessions = sessions

    def total_volume(self, session: WorkoutSession) -> float:
        """Total volume = sum(reps × weight) for all sets"""
        return sum(s.get("reps", 0) * s.get("weight", 0) for s in session.sets)

    def avg_reps_per_set(self, session: WorkoutSession) -> float:
        if not session.sets:
            return 0.0
        return np.mean([s.get("reps", 0) for s in session.sets])

    def max_weight(self, session: WorkoutSession) -> float:
        if not session.sets:
            return 0.0
        return max(s.get("weight", 0) for s in session.sets)

    def estimated_1rm(self, weight: float, reps: int) -> float:
        """Epley formula: 1RM = weight × (1 + reps/30)"""
        return weight * (1 + reps / 30)

    def weekly_volume(self, exercise: str) -> dict[str, float]:
        """Returns {week_label: total_volume}"""
        from collections import defaultdict
        weekly = defaultdict(float)
        for s in self.sessions:
            if s.exercise == exercise:
                week = s.date.strftime("%Y-W%U")
                weekly[week] += self.total_volume(s)
        return dict(sorted(weekly.items()))

    def summary(self) -> dict:
        total_sessions = len(self.sessions)
        total_time = sum(s.total_duration_min for s in self.sessions)
        total_cal = sum(s.calories_burned or 0 for s in self.sessions)
        exercises = list(set(s.exercise for s in self.sessions))
        return {
            "total_sessions": total_sessions,
            "total_duration_min": round(total_time, 2),
            "total_calories": round(total_cal, 2),
            "unique_exercises": exercises,
            "avg_session_duration_min": round(total_time / total_sessions, 2) if total_sessions else 0,
        }