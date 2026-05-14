from datetime import datetime, timedelta
from analytics.workout_analytics import WorkoutSession
from storage.storage_csv import CSVStorage

class SessionHistory:
    def __init__(self, storage: CSVStorage = None):
        self.storage = storage or CSVStorage()

    def add(self, session: WorkoutSession):
        self.storage.save(session)

    def all(self) -> list[WorkoutSession]:
        return self.storage.load_all()

    def by_exercise(self, exercise: str) -> list[WorkoutSession]:
        return [s for s in self.all() if s.exercise.lower() == exercise.lower()]

    def last_n(self, n: int) -> list[WorkoutSession]:
        sessions = sorted(self.all(), key=lambda s: s.date, reverse=True)
        return sessions[:n]

    def date_range(self, start: datetime, end: datetime) -> list[WorkoutSession]:
        return [s for s in self.all() if start <= s.date <= end]

    def last_7_days(self) -> list[WorkoutSession]:
        cutoff = datetime.now() - timedelta(days=7)
        return [s for s in self.all() if s.date >= cutoff]

    def last_30_days(self) -> list[WorkoutSession]:
        cutoff = datetime.now() - timedelta(days=30)
        return [s for s in self.all() if s.date >= cutoff]

    def exercises_list(self) -> list[str]:
        return sorted(set(s.exercise for s in self.all()))

    def delete(self, session_id: str):
        self.storage.delete(session_id)

    def stats_summary(self) -> dict:
        sessions = self.all()
        if not sessions:
            return {"total": 0}
        return {
            "total": len(sessions),
            "first_session": min(s.date for s in sessions).strftime("%Y-%m-%d"),
            "last_session": max(s.date for s in sessions).strftime("%Y-%m-%d"),
            "exercises": self.exercises_list(),
        }