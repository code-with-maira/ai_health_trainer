import numpy as np
from scipy import stats
from analytics.workout_analytics import WorkoutStatistics, WorkoutSession

class PerformanceAnalytics:
    def __init__(self, sessions: list[WorkoutSession]):
        self.stats = WorkoutStatistics(sessions)
        self.sessions = sessions

    def strength_trend(self, exercise: str) -> dict:
        """Linear regression on max weight over time"""
        filtered = [s for s in self.sessions if s.exercise == exercise]
        if len(filtered) < 2:
            return {"trend": "insufficient_data"}

        x = np.arange(len(filtered))
        y = np.array([self.stats.max_weight(s) for s in filtered])

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return {
            "slope_per_session": round(slope, 3),
            "r_squared": round(r_value ** 2, 3),
            "p_value": round(p_value, 4),
            "trend": "improving" if slope > 0 else "declining" if slope < 0 else "stable",
            "confidence": "high" if p_value < 0.05 else "low",
        }

    def volume_trend(self, exercise: str) -> dict:
        weekly = self.stats.weekly_volume(exercise)
        if len(weekly) < 2:
            return {"trend": "insufficient_data"}
        volumes = list(weekly.values())
        x = np.arange(len(volumes))
        slope, _, r, p, _ = stats.linregress(x, volumes)
        return {
            "weekly_volumes": weekly,
            "slope": round(slope, 2),
            "trend": "increasing" if slope > 0 else "decreasing",
            "r_squared": round(r ** 2, 3),
        }

    def personal_records(self) -> dict[str, dict]:
        """Best performance per exercise"""
        prs = {}
        for s in self.sessions:
            ex = s.exercise
            vol = self.stats.total_volume(s)
            mw = self.stats.max_weight(s)
            if ex not in prs or vol > prs[ex]["best_volume"]:
                prs[ex] = {
                    "best_volume": round(vol, 2),
                    "max_weight": round(mw, 2),
                    "date": s.date.strftime("%Y-%m-%d"),
                    "session_id": s.session_id,
                }
        return prs

    def consistency_score(self, target_sessions_per_week: int = 3) -> float:
        """0-100 score based on how consistently user hits weekly target"""
        from collections import Counter
        weeks = Counter(s.date.strftime("%Y-W%U") for s in self.sessions)
        if not weeks:
            return 0.0
        scores = [min(count / target_sessions_per_week, 1.0) for count in weeks.values()]
        return round(np.mean(scores) * 100, 1)