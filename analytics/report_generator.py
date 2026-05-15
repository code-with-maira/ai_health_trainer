from datetime import datetime

from analytics.workout_analytics import (
    WorkoutSession,
    WorkoutStatistics
)

from analytics.performance_analytics import (
    PerformanceAnalytics
)

from analytics.fatigue_analytics import (
    FatigueAnalytics
)

from analytics.progress_prediction import (
    ProgressPredictor
)


class ReportGenerator:

    def __init__(self, sessions=None):

        if sessions is None:
            sessions = []

        self.sessions = sessions

        self.stats = WorkoutStatistics(
            sessions
        )

        self.performance = PerformanceAnalytics(
            sessions
        )

        self.fatigue = FatigueAnalytics(
            sessions
        )

        self.predictor = ProgressPredictor(
            sessions
        )

    def generate(
            self,
            exercises: list[str] = None
    ) -> dict:

        exercises = exercises or list(
            set(
                s.exercise
                for s in self.sessions
            )
        )

        report = {

            "generated_at":
                datetime.now().isoformat(),

            "summary":
                self.stats.summary(),

            "personal_records":
                self.performance.personal_records(),

            "consistency_score":
                self.performance.consistency_score(),

            "rest_analysis":
                self.fatigue.rest_days_analysis(),

            "rpe_trend":
                self.fatigue.rpe_trend(),

            "exercises": {},
        }

        for ex in exercises:

            report["exercises"][ex] = {

                "strength_trend":
                    self.performance.strength_trend(ex),

                "volume_trend":
                    self.performance.volume_trend(ex),

                "acwr":
                    self.fatigue.acute_chronic_workload_ratio(ex),

                "plateau":
                    self.predictor.plateau_detection(ex),

                "prediction_10_sessions":
                    self.predictor.predict_max_weight(ex, 10),
            }

        return report