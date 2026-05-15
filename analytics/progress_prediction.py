import numpy as np

from sklearn.linear_model import (
    LinearRegression
)

from sklearn.preprocessing import (
    PolynomialFeatures
)

from sklearn.pipeline import (
    make_pipeline
)

from analytics.workout_analytics import (
    WorkoutSession,
    WorkoutStatistics
)


class ProgressPredictor:

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

    def predict_max_weight(
            self,
            exercise: str,
            future_sessions: int = 10
    ) -> dict:

        filtered = [

            s for s in self.sessions

            if s.exercise == exercise
        ]

        if len(filtered) < 3:

            return {
                "error":
                    "Need at least 3 sessions for prediction"
            }

        X = np.arange(
            len(filtered)
        ).reshape(-1, 1)

        y = np.array([

            self.stats.max_weight(s)

            for s in filtered
        ])

        model = make_pipeline(
            PolynomialFeatures(degree=2),
            LinearRegression()
        )

        model.fit(X, y)

        future_X = np.arange(
            len(filtered),
            len(filtered) + future_sessions
        ).reshape(-1, 1)

        predictions = model.predict(
            future_X
        )

        return {

            "exercise":
                exercise,

            "current_max":
                round(float(y[-1]), 2),

            "predicted_sessions":
                future_sessions,

            "predicted_max":
                round(float(predictions[-1]), 2),

            "predicted_values":
                [
                    round(float(p), 2)
                    for p in predictions
                ],

            "gain_estimate":
                round(
                    float(
                        predictions[-1] - y[-1]
                    ),
                    2
                ),
        }

    def predict_volume_goal(
            self,
            exercise: str,
            target_volume: float
    ) -> dict:

        filtered = [

            s for s in self.sessions

            if s.exercise == exercise
        ]

        if len(filtered) < 3:

            return {
                "error":
                    "Insufficient data"
            }

        X = np.arange(
            len(filtered)
        ).reshape(-1, 1)

        y = np.array([

            self.stats.total_volume(s)

            for s in filtered
        ])

        model = LinearRegression().fit(
            X,
            y
        )

        if model.coef_[0] <= 0:

            return {

                "sessions_needed":
                    None,

                "message":
                    "Volume is not increasing"
            }

        sessions_needed = int(
            np.ceil(
                (
                    target_volume -
                    model.intercept_
                ) / model.coef_[0]
            )
        )

        remaining = max(
            0,
            sessions_needed - len(filtered)
        )

        return {

            "target_volume":
                target_volume,

            "current_avg_volume":
                round(float(np.mean(y)), 2),

            "sessions_needed":
                sessions_needed,

            "remaining_sessions":
                remaining,
        }

    def plateau_detection(
            self,
            exercise: str,
            window: int = 5
    ) -> dict:

        filtered = [

            s for s in self.sessions

            if s.exercise == exercise
        ]

        if len(filtered) < window:

            return {

                "plateau":
                    False,

                "reason":
                    "Insufficient data"
            }

        recent = [

            self.stats.max_weight(s)

            for s in filtered[-window:]
        ]

        variance = np.var(recent)

        mean = np.mean(recent)

        cv = (

            np.std(recent) /
            mean * 100

        ) if mean > 0 else 0

        plateau = cv < 2.0

        return {

            "plateau_detected":
                plateau,

            "coefficient_of_variation":
                round(cv, 2),

            "recent_weights":
                recent,

            "recommendation":
                "Deload or change stimulus"
                if plateau else
                "Progress ongoing",
        }