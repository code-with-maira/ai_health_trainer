import numpy as np
from datetime import datetime, timedelta
from analytics.workout_analytics import WorkoutSession, WorkoutStatistics

class FatigueAnalytics:
    def __init__(self, sessions: list[WorkoutSession]):
        self.sessions = sorted(sessions, key=lambda s: s.date)
        self.stats = WorkoutStatistics(sessions)

    def acute_chronic_workload_ratio(self, exercise: str,
                                      acute_days=7, chronic_days=28) -> dict:
        """
        ACWR = acute load / chronic load
        < 0.8  → undertraining
        0.8-1.3 → optimal
        > 1.5  → high injury risk
        """
        now = self.sessions[-1].date if self.sessions else datetime.now()
        acute_cutoff = now - timedelta(days=acute_days)
        chronic_cutoff = now - timedelta(days=chronic_days)

        acute_sessions = [s for s in self.sessions
                          if s.exercise == exercise and s.date >= acute_cutoff]
        chronic_sessions = [s for s in self.sessions
                            if s.exercise == exercise and s.date >= chronic_cutoff]

        acute_load = np.mean([self.stats.total_volume(s) for s in acute_sessions]) if acute_sessions else 0
        chronic_load = np.mean([self.stats.total_volume(s) for s in chronic_sessions]) if chronic_sessions else 0

        acwr = acute_load / chronic_load if chronic_load > 0 else None
        status = (
            "undertraining" if acwr and acwr < 0.8 else
            "optimal" if acwr and acwr <= 1.3 else
            "high_risk" if acwr and acwr > 1.5 else
            "moderate_risk"
        )
        return {
            "acute_load": round(acute_load, 2),
            "chronic_load": round(chronic_load, 2),
            "acwr": round(acwr, 3) if acwr else None,
            "status": status,
        }

    def rest_days_analysis(self) -> dict:
        """Analyze gaps between sessions"""
        if len(self.sessions) < 2:
            return {"avg_rest_days": None}
        gaps = [(self.sessions[i].date - self.sessions[i-1].date).days
                for i in range(1, len(self.sessions))]
        return {
            "avg_rest_days": round(np.mean(gaps), 1),
            "min_rest_days": int(np.min(gaps)),
            "max_rest_days": int(np.max(gaps)),
            "overtraining_flags": sum(1 for g in gaps if g == 0),
            "undertraining_flags": sum(1 for g in gaps if g > 7),
        }

    def rpe_trend(self) -> dict:
        """Track perceived exertion from notes (expects 'RPE:X' in notes)"""
        import re
        rpe_data = []
        for s in self.sessions:
            match = re.search(r"RPE[:\s](\d+(\.\d+)?)", s.notes, re.IGNORECASE)
            if match:
                rpe_data.append({
                    "date": s.date.strftime("%Y-%m-%d"),
                    "rpe": float(match.group(1))
                })
        if not rpe_data:
            return {"rpe_data": [], "avg_rpe": None}
        avg = np.mean([r["rpe"] for r in rpe_data])
        return {"rpe_data": rpe_data, "avg_rpe": round(avg, 2)}