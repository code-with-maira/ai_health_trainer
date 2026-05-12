# =========================================================
# ai/models/fatigue_model.py
# =========================================================

import numpy as np
import pickle
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class FatigueModel:
    """
    AI Fatigue Detection Model

    Detects:
    - fresh
    - mild_fatigue
    - high_fatigue
    - exhausted
    """

    LEVELS = [
        "fresh",
        "mild_fatigue",
        "high_fatigue",
        "exhausted"
    ]

    def __init__(self):

        self.model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

        self.encoder = LabelEncoder()

        self.is_trained = False

        self.feature_count = 6

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================

    def extract_features(self, movement_data):

        movement_data = np.array(
            movement_data,
            dtype=np.float32
        )

        if movement_data.ndim < 2:
            raise ValueError(
                "movement_data must be 2D"
            )

        # ---------------------------------------------
        # Motion derivatives
        # ---------------------------------------------

        speeds = np.diff(
            movement_data,
            axis=0
        )

        accelerations = np.diff(
            speeds,
            axis=0
        )

        # ---------------------------------------------
        # Features
        # ---------------------------------------------

        avg_speed = np.mean(
            np.abs(speeds)
        )

        speed_std = np.std(
            speeds
        )

        motion_range = (
            np.max(movement_data) -
            np.min(movement_data)
        )

        smoothness = np.mean(
            np.abs(accelerations)
        )

        fatigue_shake = np.std(
            accelerations
        )

        movement_stability = np.mean(
            np.var(movement_data, axis=0)
        )

        features = [

            avg_speed,

            speed_std,

            motion_range,

            smoothness,

            fatigue_shake,

            movement_stability
        ]

        return np.array(
            features,
            dtype=np.float32
        )

    # =====================================================
    # TRAIN
    # =====================================================

    def train(self, X, y):

        X = np.array(
            X,
            dtype=np.float32
        )

        encoded_y = self.encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            encoded_y,
            test_size=0.2,
            random_state=42
        )

        self.model.fit(
            X_train,
            y_train
        )

        preds = self.model.predict(X_test)

        acc = accuracy_score(
            y_test,
            preds
        )

        print(f"[FatigueModel] Accuracy: {acc:.2f}")

        self.is_trained = True

    # =====================================================
    # PREDICT
    # =====================================================

    def predict_fatigue(self, movement_data):

        if not self.is_trained:
            raise RuntimeError(
                "Model not trained"
            )

        features = self.extract_features(
            movement_data
        )

        features = features.reshape(1, -1)

        encoded_pred = self.model.predict(
            features
        )[0]

        probs = self.model.predict_proba(
            features
        )[0]

        fatigue = self.encoder.inverse_transform(
            [encoded_pred]
        )[0]

        confidence = float(np.max(probs))

        return {

            "fatigue_level": fatigue,

            "confidence": round(
                confidence,
                3
            ),

            "recommendation": self.get_recommendation(
                fatigue
            ),

            "all_probabilities": {

                self.encoder.inverse_transform([i])[0]:
                round(float(p), 3)

                for i, p in enumerate(probs)
            }
        }

    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    def get_recommendation(self, fatigue):

        recommendations = {

            "fresh":
                "Energy level looks good.",

            "mild_fatigue":
                "Take a short rest soon.",

            "high_fatigue":
                "Reduce intensity and slow down.",

            "exhausted":
                "Stop workout and recover."
        }

        return recommendations.get(
            fatigue,
            ""
        )

    # =====================================================
    # SAVE
    # =====================================================

    def save(
        self,
        path="saved_models/fatigue_model.pkl"
    ):

        os.makedirs(
            "saved_models",
            exist_ok=True
        )

        with open(path, "wb") as f:

            pickle.dump({

                "model": self.model,

                "encoder": self.encoder

            }, f)

        print(f"Saved -> {path}")

    # =====================================================
    # LOAD
    # =====================================================

    def load(
        self,
        path="saved_models/fatigue_model.pkl"
    ):

        with open(path, "rb") as f:

            data = pickle.load(f)

        self.model = data["model"]

        self.encoder = data["encoder"]

        self.is_trained = True

        print(f"Loaded <- {path}")


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    model = FatigueModel()

    # ---------------------------------------------
    # Dummy training data
    # ---------------------------------------------

    X = np.random.rand(
        400,
        6
    )

    y = np.random.choice(
        model.LEVELS,
        400
    )

    model.train(X, y)

    # ---------------------------------------------
    # Fake movement sequence
    # ---------------------------------------------

    movement_data = np.random.rand(
        30,
        10
    )

    result = model.predict_fatigue(
        movement_data
    )

    print("\nPrediction:")
    print(result)

    model.save()