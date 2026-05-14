import csv
import os
from datetime import datetime
from analytics.workout_analytics import WorkoutSession

CSV_PATH = "data/sessions.csv"

FIELDS = [
    "session_id", "date", "exercise", "total_duration_min",
    "heart_rate_avg", "calories_burned", "notes", "sets_json"
]

class CSVStorage:
    def __init__(self, path: str = CSV_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            self._init_file()

    def _init_file(self):
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()

    def save(self, session: WorkoutSession):
        import json
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writerow({
                "session_id": session.session_id,
                "date": session.date.isoformat(),
                "exercise": session.exercise,
                "total_duration_min": session.total_duration_min,
                "heart_rate_avg": session.heart_rate_avg or "",
                "calories_burned": session.calories_burned or "",
                "notes": session.notes,
                "sets_json": json.dumps(session.sets),
            })

    def load_all(self) -> list[WorkoutSession]:
        import json
        sessions = []
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sessions.append(WorkoutSession(
                    session_id=row["session_id"],
                    date=datetime.fromisoformat(row["date"]),
                    exercise=row["exercise"],
                    total_duration_min=float(row["total_duration_min"]),
                    heart_rate_avg=float(row["heart_rate_avg"]) if row["heart_rate_avg"] else None,
                    calories_burned=float(row["calories_burned"]) if row["calories_burned"] else None,
                    notes=row["notes"],
                    sets=json.loads(row["sets_json"]),
                ))
        return sessions

    def delete(self, session_id: str):
        rows = []
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r["session_id"] != session_id]
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)