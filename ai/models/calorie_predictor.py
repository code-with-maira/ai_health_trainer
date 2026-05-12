# =========================================================
# ai/models/calorie_predictor.py
# =========================================================

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)


class CaloriePredictor:

    def __init__(self):

        self.features = [
            "age",
            "weight",
            "height",
            "duration_mins",
            "heart_rate",
            "gender",
        ]

        self.scaler = StandardScaler()

        self.model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        )

        self.is_trained = False

    # =====================================================
    # TRAIN
    # =====================================================

    def train(self, df: pd.DataFrame):

        print("\n[INFO] Training calorie predictor...")

        # Validate columns
        required = self.features + ["calories"]

        for col in required:

            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        X = df[self.features].values.astype(np.float32)

        y = df["calories"].values.astype(np.float32)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        # Scale
        X_train = self.scaler.fit_transform(X_train)

        X_test = self.scaler.transform(X_test)

        # Train
        self.model.fit(X_train, y_train)

        self.is_trained = True

        # Evaluate
        predictions = self.model.predict(X_test)

        r2 = r2_score(y_test, predictions)

        mae = mean_absolute_error(y_test, predictions)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        print(f"\n✅ R2 Score : {r2:.3f}")
        print(f"✅ MAE      : {mae:.2f}")
        print(f"✅ RMSE     : {rmse:.2f}")

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(
        self,
        age,
        weight,
        height,
        duration,
        heart_rate,
        gender=1,
    ):

        if not self.is_trained:
            raise RuntimeError("Model not trained")

        data = np.array([
            age,
            weight,
            height,
            duration,
            heart_rate,
            gender,
        ], dtype=np.float32).reshape(1, -1)

        data_scaled = self.scaler.transform(data)

        calories = self.model.predict(data_scaled)[0]

        return round(float(calories), 2)

    # =====================================================
    # SAVE
    # =====================================================

    def save(
        self,
        path="deep_learning/weights/calorie_model.pkl",
    ):

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        with open(path, "wb") as f:

            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "features": self.features,
                },
                f,
            )

        print(f"\n✅ Model saved: {path}")

    # =====================================================
    # LOAD
    # =====================================================

    def load(
        self,
        path="deep_learning/weights/calorie_model.pkl",
    ):

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with open(path, "rb") as f:

            data = pickle.load(f)

        self.model = data["model"]

        self.scaler = data["scaler"]

        self.features = data["features"]

        self.is_trained = True

        print(f"\n✅ Model loaded: {path}")

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    def feature_importance(self):

        if not self.is_trained:
            raise RuntimeError("Model not trained")

        importance = self.model.feature_importances_

        scores = dict(
            sorted(
                zip(self.features, importance),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        print("\nFeature Importance:\n")

        for k, v in scores.items():

            print(f"{k:<18} : {v:.4f}")


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    predictor = CaloriePredictor()

    # =====================================================
    # Dummy Dataset
    # =====================================================

    df = pd.DataFrame({

        "age": np.random.randint(18, 60, 1000),

        "weight": np.random.randint(45, 110, 1000),

        "height": np.random.randint(150, 195, 1000),

        "duration_mins": np.random.randint(5, 90, 1000),

        "heart_rate": np.random.randint(60, 190, 1000),

        "gender": np.random.randint(0, 2, 1000),

        "calories": np.random.randint(50, 900, 1000),
    })

    # =====================================================
    # TRAIN
    # =====================================================

    predictor.train(df)

    # =====================================================
    # TEST PREDICTION
    # =====================================================

    result = predictor.predict(
        age=25,
        weight=70,
        height=175,
        duration=30,
        heart_rate=145,
        gender=1,
    )

    print(f"\n🔥 Predicted Calories Burned: {result}")

    # =====================================================
    # SAVE
    # =====================================================

    predictor.save()

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    predictor.feature_importance()