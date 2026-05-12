# =========================================================
# ai/models/exercise_classifier.py
# Advanced MediaPipe 33-Landmark Exercise Classifier
# =========================================================

import os
import pickle
import numpy as np
import pandas as pd

from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


class ExerciseClassifier:
    """
    Advanced Exercise Classifier

    Uses:
    - MediaPipe 33 landmarks
    - x,y,z coordinates
    - 99 features total

    Feature shape:
    33 landmarks × 3 coordinates = 99 features
    """

    EXERCISE_CLASSES = [
        "squat",
        "pushup",
        "plank",
        "jumping_jack",
        "lunge",
        "deadlift",
    ]

    def __init__(self):

        self.num_landmarks = 33
        self.num_features = self.num_landmarks * 3   # 99

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

        self.label_encoder = LabelEncoder()

        self.is_trained = False

    # =====================================================
    # LOAD DATASET
    # =====================================================

    def load_csv_dataset(self, csv_path: str):

        """
        CSV format:

        feature_0 ... feature_98 label
        """

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if "label" not in df.columns:
            raise ValueError("CSV must contain 'label' column")

        X = df.drop("label", axis=1).values.astype(np.float32)
        y = df["label"].values

        if X.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, "
                f"got {X.shape[1]}"
            )

        return X, y

    # =====================================================
    # TRAIN
    # =====================================================

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
    ):

        print("\n[INFO] Encoding labels...")

        y_encoded = self.label_encoder.fit_transform(y)

        print("[INFO] Splitting dataset...")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=test_size,
            random_state=42,
            stratify=y_encoded,
        )

        print("[INFO] Training RandomForest model...")

        self.model.fit(X_train, y_train)

        self.is_trained = True

        print("[INFO] Evaluating model...")

        predictions = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")

        print("\nClassification Report:\n")

        print(
            classification_report(
                y_test,
                predictions,
                target_names=self.label_encoder.classes_,
            )
        )

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self, features):

        """
        Input:
        shape = (99,)
        """

        if not self.is_trained:
            raise RuntimeError("Model not trained")

        features = np.array(features, dtype=np.float32)

        if len(features) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, "
                f"got {len(features)}"
            )

        features = features.reshape(1, -1)

        prediction = self.model.predict(features)[0]

        label = self.label_encoder.inverse_transform([prediction])[0]

        return label

    # =====================================================
    # PREDICT WITH CONFIDENCE
    # =====================================================

    def predict_with_confidence(self, features):

        if not self.is_trained:
            raise RuntimeError("Model not trained")

        features = np.array(features, dtype=np.float32)

        if len(features) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features"
            )

        features = features.reshape(1, -1)

        probabilities = self.model.predict_proba(features)[0]

        best_idx = np.argmax(probabilities)

        prediction = self.label_encoder.inverse_transform([best_idx])[0]

        confidence = float(probabilities[best_idx])

        all_scores = {}

        for i, prob in enumerate(probabilities):

            class_name = self.label_encoder.inverse_transform([i])[0]

            all_scores[class_name] = round(float(prob), 4)

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "scores": dict(
                sorted(
                    all_scores.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        }

    # =====================================================
    # SAVE MODEL
    # =====================================================

    def save(
        self,
        model_path: str = "deep_learning/weights/exercise_model.pkl",
    ):

        os.makedirs(
            os.path.dirname(model_path),
            exist_ok=True,
        )

        data = {
            "model": self.model,
            "label_encoder": self.label_encoder,
        }

        with open(model_path, "wb") as f:
            pickle.dump(data, f)

        print(f"\n✅ Model saved: {model_path}")

    # =====================================================
    # LOAD MODEL
    # =====================================================

    def load(
        self,
        model_path: str = "deep_learning/weights/exercise_model.pkl",
    ):

        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.label_encoder = data["label_encoder"]

        self.is_trained = True

        print(f"\n✅ Model loaded: {model_path}")

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    def feature_importance(self):

        if not self.is_trained:
            raise RuntimeError("Model not trained")

        importances = self.model.feature_importances_

        top = np.argsort(importances)[::-1][:10]

        print("\nTop Important Features:\n")

        for idx in top:

            print(
                f"Feature_{idx:<3} : "
                f"{importances[idx]:.5f}"
            )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    clf = ExerciseClassifier()

    # =====================================================
    # DUMMY DATASET
    # Replace later with real CSV dataset
    # =====================================================

    samples = 1000

    X = np.random.rand(samples, 99).astype(np.float32)

    y = np.random.choice(
        clf.EXERCISE_CLASSES,
        samples,
    )

    # =====================================================
    # TRAIN
    # =====================================================

    clf.train(X, y)

    # =====================================================
    # TEST PREDICTION
    # =====================================================

    test_features = np.random.rand(99)

    result = clf.predict_with_confidence(test_features)

    print("\nPrediction Result:\n")

    print(result)

    # =====================================================
    # SAVE MODEL
    # =====================================================

    clf.save()

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    clf.feature_importance()