# =========================================================
# ai/models/injury_risk_model.py
# =========================================================

import numpy as np
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class InjuryRiskModel:
    """
    AI Injury Risk Prediction Model

    Predicts:
    - low
    - medium
    - high injury risk
    """

    RISK_LEVELS = [
        "low",
        "medium",
        "high"
    ]

    def __init__(self):

        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=10,
            class_weight="balanced"
        )

        self.encoder = LabelEncoder()

        self.is_trained = False

        # Expected features:
        # 5 joint angles +
        # movement speed +
        # fatigue +
        # form score
        self.feature_count = 8

    # =====================================================
    # VALIDATE FEATURES
    # =====================================================

    def validate_features(self, features):

        arr = np.array(features, dtype=np.float32)

        if arr.ndim != 1:
            raise ValueError(
                f"1D features required, got {arr.shape}"
            )

        if len(arr) != self.feature_count:
            raise ValueError(
                f"Expected {self.feature_count} features, got {len(arr)}"
            )

        if np.any(np.isnan(arr)):
            raise ValueError("NaN values detected")

        if np.any(np.isinf(arr)):
            raise ValueError("Inf values detected")

        return arr.reshape(1, -1)

    # =====================================================
    # TRAIN
    # =====================================================

    def train(self, X, y):

        X = np.array(X, dtype=np.float32)

        encoded_y = self.encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            encoded_y,
            test_size=0.2,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        print(f"[InjuryRiskModel] Accuracy: {acc:.2f}")

        self.is_trained = True

    # =====================================================
    # PREDICT
    # =====================================================

    def predict_risk(
        self,
        joint_angles,
        speed,
        fatigue,
        form_score
    ):

        if not self.is_trained:
            raise RuntimeError("Model not trained")

        features = (
            list(joint_angles) +
            [speed, fatigue, form_score]
        )

        arr = self.validate_features(features)

        encoded_pred = self.model.predict(arr)[0]

        probs = self.model.predict_proba(arr)[0]

        confidence = float(np.max(probs))

        risk = self.encoder.inverse_transform(
            [encoded_pred]
        )[0]

        return {
            "risk": risk,
            "confidence": round(confidence * 100, 1),
            "advice": self.get_advice(risk),
            "all_probabilities": {
                self.encoder.inverse_transform([i])[0]:
                round(float(p), 3)
                for i, p in enumerate(probs)
            }
        }

    # =====================================================
    # ADVICE
    # =====================================================

    def get_advice(self, risk):

        advice = {
            "low":
                "Good form detected — continue safely.",

            "medium":
                "Reduce speed and focus on posture.",

            "high":
                "High injury risk detected — stop and correct form."
        }

        return advice.get(risk, "")

    # =====================================================
    # SAVE
    # =====================================================

    def save(self, path="saved_models/injury_risk_model.pkl"):

        os.makedirs("saved_models", exist_ok=True)

        with open(path, "wb") as f:

            pickle.dump({
                "model": self.model,
                "encoder": self.encoder,
            }, f)

        print(f"Saved -> {path}")

    # =====================================================
    # LOAD
    # =====================================================

    def load(self, path="saved_models/injury_risk_model.pkl"):

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

    model = InjuryRiskModel()

    # Dummy dataset
    X = np.random.rand(500, 8)

    y = np.random.choice(
        model.RISK_LEVELS,
        500
    )

    model.train(X, y)

    # Example prediction
    result = model.predict_risk(
        joint_angles=[0.5, 0.3, 0.8, 0.6, 0.4],
        speed=0.7,
        fatigue=0.3,
        form_score=0.8
    )

    print("\nPrediction:")
    print(result)

    model.save()