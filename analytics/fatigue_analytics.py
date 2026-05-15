import numpy as np
from datetime import datetime, timedelta
from analytics.workout_analytics import WorkoutSession, WorkoutStatistics


class FatigueAnalytics:

    def __init__(self, sessions=None):

        if sessions is None:
            sessions = []

        self.sessions = sorted(
            sessions,
            key=lambda s: s.date
        )

        self.stats = WorkoutStatistics(
            sessions
        )

    def acute_chronic_workload_ratio(
            self,
            exercise: str,
            acute_days=7,
            chronic_days=28):

        now = (
            self.sessions[-1].date
            if self.sessions
            else datetime.now()
        )

        acute_cutoff = now - timedelta(days=acute_days)

        chronic_cutoff = now - timedelta(days=chronic_days)

        acute_sessions = [
            s for s in self.sessions
            if s.exercise == exercise
            and s.date >= acute_cutoff
        ]

        chronic_sessions = [
            s for s in self.sessions
            if s.exercise == exercise
            and s.date >= chronic_cutoff
        ]

        acute_load = (
            np.mean([
                self.stats.total_volume(s)
                for s in acute_sessions
            ])
            if acute_sessions else 0
        )

        chronic_load = (
            np.mean([
                self.stats.total_volume(s)
                for s in chronic_sessions
            ])
            if chronic_sessions else 0
        )

        acwr = (
            acute_load / chronic_load
            if chronic_load > 0
            else None
        )

        status = (
            "undertraining"
            if acwr and acwr < 0.8 else
            "optimal"
            if acwr and acwr <= 1.3 else
            "high_risk"
            if acwr and acwr > 1.5 else
            "moderate_risk"
        )

        return {
            "acute_load": round(acute_load, 2),
            "chronic_load": round(chronic_load, 2),
            "acwr": round(acwr, 3) if acwr else None,
            "status": status,
        }