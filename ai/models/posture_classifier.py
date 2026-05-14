# =========================================================
# ai/models/posture_classifier.py
# =========================================================

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os


class PostureClassifier:
    """
    AI posture classifier using body landmark angles.
    Detects:
    - good_posture
    - bad_posture
    - slouching
    - leaning
    """

    LABELS = [
        "good_posture",
        "bad_posture",
        "slouching",
        "leaning"
    ]

    def __init__(self):

        # Labels access
        self.labels = self.LABELS

        # StandardScaler + SVM pipeline
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                probability=True,
                C=10,
                gamma="scale"
            ))
        ])

        self.is_trained = False

    # =====================================================
    # ANGLE CALCULATION
    # =====================================================

    def calculate_angle(self, a, b, c):
        """
        Calculate angle between 3 points.
        """

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        denominator = (
            np.linalg.norm(ba) *
            np.linalg.norm(bc)
        ) + 1e-8

        cosine = np.dot(ba, bc) / denominator

        cosine = np.clip(cosine, -1.0, 1.0)

        angle = np.degrees(np.arccos(cosine))

        return float(angle)

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================

    def extract_features(self, keypoints):
        """
        Extract posture angles from landmarks.
        """

        keypoints = np.array(keypoints)

        if len(keypoints) < 17:
            raise ValueError(
                "At least 17 keypoints required"
            )

        features = []

        # Left shoulder-elbow-wrist
        features.append(
            self.calculate_angle(
                keypoints[5],
                keypoints[7],
                keypoints[9]
            )
        )

        # Right shoulder-elbow-wrist
        features.append(
            self.calculate_angle(
                keypoints[6],
                keypoints[8],
                keypoints[10]
            )
        )

        # Left hip-knee-ankle
        features.append(
            self.calculate_angle(
                keypoints[11],
                keypoints[13],
                keypoints[15]
            )
        )

        # Right hip-knee-ankle
        features.append(
            self.calculate_angle(
                keypoints[12],
                keypoints[14],
                keypoints[16]
            )
        )

        # Shoulder alignment
        shoulder_diff = abs(
            keypoints[5][1] - keypoints[6][1]
        )
        features.append(shoulder_diff)

        # Hip alignment
        hip_diff = abs(
            keypoints[11][1] - keypoints[12][1]
        )
        features.append(hip_diff)

        return np.array(features)

    # =====================================================
    # TRAIN
    # =====================================================

    def train(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        print(f"[PostureClassifier] Accuracy: {acc:.2f}")

        self.is_trained = True

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self, keypoints):

        if not self.is_trained:
            raise RuntimeError("Model not trained")

        features = self.extract_features(keypoints)

        pred = self.model.predict([features])[0]

        probs = self.model.predict_proba([features])[0]

        confidence = float(np.max(probs))

        return {
            "posture": pred,
            "confidence": round(confidence, 3)
        }

    # =====================================================
    # SAVE
    # =====================================================

    def save(self, path="saved_models/posture_classifier.pkl"):

        os.makedirs("saved_models", exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.model, f)

        print(f"Saved model -> {path}")

    # =====================================================
    # LOAD
    # =====================================================

    def load(self, path="saved_models/posture_classifier.pkl"):

        with open(path, "rb") as f:
            self.model = pickle.load(f)

        self.is_trained = True

        print(f"Loaded model <- {path}")


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    clf = PostureClassifier()

    # Dummy dataset
    X = np.random.rand(300, 6)

    y = np.random.choice(
        clf.labels,
        300
    )

    clf.train(X, y)

    clf.save()

    # Fake 17 keypoints
    dummy_pose = np.random.rand(17, 2)

    result = clf.predict(dummy_pose)

    print(result)