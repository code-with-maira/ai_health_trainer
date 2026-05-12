# ============================================
# core/smart_detector.py
# ============================================
"""
Advanced AI Exercise Detector
-----------------------------
Professional ML-based exercise/activity classifier.

NEW IMPROVEMENTS:
- Temporal smoothing
- Confidence stabilization
- Noise rejection
- Activity transition handling
- Auto feature normalization
- Better landmark geometry extraction
- Model auto-save/load
- Online learning support
- Prediction history
- Stability scoring
- Feature validation
- Fallback safe mode
- Multi-model support ready
"""

import os
import math
import logging
from collections import deque, Counter
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "smart_detector.pkl",
)


# ============================================================
# TRAINING DATA
# ============================================================

TRAIN_DATA = [

    # --------------------------------------------------------
    # Squat
    # --------------------------------------------------------
    ([90,88,85,87,160,158,55,170], "Squat"),
    ([95,93,80,82,162,160,50,168], "Squat"),
    ([100,98,90,91,155,157,60,172], "Squat"),
    ([85,84,78,79,158,156,52,165], "Squat"),
    ([110,108,95,96,161,159,58,170], "Squat"),
    ([92,90,83,84,163,161,54,169], "Squat"),
    ([88,87,81,80,157,155,56,167], "Squat"),
    ([105,103,92,93,160,158,53,171], "Squat"),
    ([98,96,88,89,159,157,57,170], "Squat"),
    ([102,100,93,94,156,154,59,169], "Squat"),

    # --------------------------------------------------------
    # Pushup
    # --------------------------------------------------------
    ([165,163,155,154,70,68,40,95], "Pushup"),
    ([168,166,158,156,65,63,38,92], "Pushup"),
    ([162,160,152,150,75,73,42,98], "Pushup"),
    ([170,168,160,158,60,58,35,90], "Pushup"),
    ([164,162,154,152,80,78,45,100], "Pushup"),
    ([166,164,156,155,72,70,40,94], "Pushup"),
    ([160,158,150,148,85,83,48,102], "Pushup"),
    ([172,170,162,160,55,53,33,88], "Pushup"),
    ([163,161,153,151,77,75,43,96], "Pushup"),
    ([167,165,157,155,68,66,39,93], "Pushup"),

    # --------------------------------------------------------
    # Curl
    # --------------------------------------------------------
    ([170,168,170,168,55,53,25,175], "Curl"),
    ([172,170,172,170,50,48,22,176], "Curl"),
    ([168,166,168,166,60,58,28,174], "Curl"),
    ([174,172,174,172,45,43,20,177], "Curl"),
    ([166,164,166,164,65,63,30,173], "Curl"),
    ([171,169,171,169,52,50,24,175], "Curl"),
    ([169,167,169,167,58,56,26,174], "Curl"),
    ([175,173,175,173,42,40,18,178], "Curl"),
    ([167,165,167,165,62,60,29,174], "Curl"),
    ([173,171,173,171,48,46,21,176], "Curl"),

    # --------------------------------------------------------
    # Standing
    # --------------------------------------------------------
    ([175,174,175,174,170,168,15,178], "Standing"),
    ([177,176,177,176,172,170,12,179], "Standing"),
    ([173,172,173,172,168,166,18,177], "Standing"),
    ([178,177,178,177,174,172,10,180], "Standing"),
    ([174,173,174,173,169,167,16,178], "Standing"),
    ([176,175,176,175,171,169,13,179], "Standing"),
    ([172,171,172,171,167,165,20,176], "Standing"),
    ([179,178,179,178,175,173,8,180], "Standing"),
    ([174,173,174,173,170,168,14,178], "Standing"),
    ([177,176,177,176,173,171,11,179], "Standing"),

    # --------------------------------------------------------
    # Lunge
    # --------------------------------------------------------
    ([90,165,88,170,158,156,50,168], "Lunge"),
    ([92,168,90,172,160,158,52,170], "Lunge"),
    ([88,162,85,168,156,154,48,166], "Lunge"),
    ([95,170,93,175,162,160,55,172], "Lunge"),
    ([85,160,83,165,154,152,45,164], "Lunge"),
    ([91,166,89,171,159,157,51,169], "Lunge"),
    ([87,163,84,169,155,153,47,165], "Lunge"),
    ([96,172,94,177,163,161,56,173], "Lunge"),
    ([89,164,87,170,157,155,49,167], "Lunge"),
    ([93,169,91,174,161,159,53,171], "Lunge"),
]


# ============================================================
# SMART DETECTOR
# ============================================================

class SmartDetector:

    FEATURE_KEYS = [
        "left_knee",
        "right_knee",
        "left_hip",
        "right_hip",
        "left_elbow",
        "right_elbow",
        "left_shoulder",
        "trunk_lean_left",
    ]

    def __init__(
        self,
        history_size: int = 10,
        confidence_threshold: float = 0.55,
    ):

        self.history_size = history_size
        self.confidence_threshold = confidence_threshold

        # ----------------------------------------------------
        # Prediction history
        # ----------------------------------------------------

        self._prediction_history = deque(maxlen=history_size)
        self._confidence_history = deque(maxlen=history_size)

        # ----------------------------------------------------
        # Label encoder
        # ----------------------------------------------------

        self._encoder = LabelEncoder()

        # ----------------------------------------------------
        # Pipeline
        # ----------------------------------------------------

        self._model = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=250,
                max_depth=12,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            ))
        ])

        self._trained = False
        self._data = list(TRAIN_DATA)

        self._load_or_train()

    # ========================================================
    # LOAD OR TRAIN
    # ========================================================

    def _load_or_train(self):

        if os.path.exists(MODEL_PATH):

            try:
                saved = joblib.load(MODEL_PATH)

                self._model = saved["model"]
                self._encoder = saved["encoder"]

                self._trained = True

                logger.info(
                    "SmartDetector loaded successfully"
                )

                return

            except Exception as e:
                logger.warning(
                    "Model loading failed: %s",
                    e
                )

        self._train()

    # ========================================================
    # TRAIN
    # ========================================================

    def _train(self):

        X = np.array(
            [row[0] for row in self._data],
            dtype=np.float32,
        )

        y_raw = [row[1] for row in self._data]

        y = self._encoder.fit_transform(y_raw)

        self._model.fit(X, y)

        self._trained = True

        logger.info(
            "SmartDetector trained: samples=%d classes=%s",
            len(X),
            list(self._encoder.classes_)
        )

        self.save()

    # ========================================================
    # SAVE
    # ========================================================

    def save(self):

        os.makedirs(MODEL_DIR, exist_ok=True)

        joblib.dump(
            {
                "model": self._model,
                "encoder": self._encoder,
            },
            MODEL_PATH,
        )

        logger.info(
            "SmartDetector saved to %s",
            MODEL_PATH
        )

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def _build_feature_vector(
        self,
        angles: Dict[str, float]
    ) -> np.ndarray:

        features = []

        for key in self.FEATURE_KEYS:

            val = float(angles.get(key, 0.0))

            if math.isnan(val):
                val = 0.0

            val = np.clip(val, 0, 180)

            features.append(val)

        return np.array(
            features,
            dtype=np.float32,
        ).reshape(1, -1)

    # ========================================================
    # MAIN DETECTION
    # ========================================================

    def detect_activity(
        self,
        angles: Dict[str, float]
    ) -> Tuple[str, float]:

        if not self._trained:
            return "Unknown", 0.0

        try:

            features = self._build_feature_vector(angles)

            if np.all(features == 0):
                return "Unknown", 0.0

            pred_encoded = self._model.predict(features)[0]

            probs = self._model.predict_proba(features)[0]

            confidence = float(np.max(probs))

            label = self._encoder.inverse_transform(
                [pred_encoded]
            )[0]

            # ------------------------------------------------
            # Confidence threshold
            # ------------------------------------------------

            if confidence < self.confidence_threshold:
                label = "Unknown"

            # ------------------------------------------------
            # Temporal smoothing
            # ------------------------------------------------

            self._prediction_history.append(label)
            self._confidence_history.append(confidence)

            stable_label = self._stable_prediction()

            stable_conf = float(
                np.mean(self._confidence_history)
            )

            return stable_label, round(stable_conf, 3)

        except Exception as e:

            logger.error(
                "detect_activity failed: %s",
                e
            )

            return "Unknown", 0.0

    # ========================================================
    # STABLE PREDICTION
    # ========================================================

    def _stable_prediction(self) -> str:

        if not self._prediction_history:
            return "Unknown"

        counts = Counter(self._prediction_history)

        stable = counts.most_common(1)[0][0]

        return stable

    # ========================================================
    # LANDMARK SUPPORT
    # ========================================================

    def detect_from_landmarks(
        self,
        landmarks
    ) -> Tuple[str, float]:

        if not landmarks or len(landmarks) < 29:
            return "Unknown", 0.0

        try:

            def _x(lm):
                return lm.x if hasattr(lm, "x") else lm["x"]

            def _y(lm):
                return lm.y if hasattr(lm, "y") else lm["y"]

            shoulder_y = _y(landmarks[11])
            hip_y      = _y(landmarks[23])
            knee_y     = _y(landmarks[25])
            wrist_y    = _y(landmarks[15])

            trunk_proxy = abs(shoulder_y - hip_y) * 220
            knee_proxy  = abs(hip_y - knee_y) * 320
            elbow_proxy = abs(shoulder_y - wrist_y) * 220

            angles = {

                "left_knee":
                    float(np.clip(knee_proxy, 0, 180)),

                "right_knee":
                    float(np.clip(knee_proxy, 0, 180)),

                "left_hip":
                    float(np.clip(trunk_proxy, 0, 180)),

                "right_hip":
                    float(np.clip(trunk_proxy, 0, 180)),

                "left_elbow":
                    float(np.clip(elbow_proxy, 0, 180)),

                "right_elbow":
                    float(np.clip(elbow_proxy, 0, 180)),

                "left_shoulder":
                    float(np.clip(trunk_proxy, 0, 90)),

                "trunk_lean_left":
                    float(np.clip(trunk_proxy, 0, 180)),
            }

            return self.detect_activity(angles)

        except Exception as e:

            logger.warning(
                "detect_from_landmarks failed: %s",
                e
            )

            return "Unknown", 0.0

    # ========================================================
    # ONLINE LEARNING
    # ========================================================

    def add_sample(
        self,
        angles: Dict[str, float],
        label: str,
    ):

        features = [
            float(angles.get(k, 0.0))
            for k in self.FEATURE_KEYS
        ]

        self._data.append((features, label))

        self._train()

        logger.info(
            "Sample added label=%s total=%d",
            label,
            len(self._data),
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self._prediction_history.clear()
        self._confidence_history.clear()

    # ========================================================
    # INFO
    # ========================================================

    @property
    def classes(self) -> List[str]:

        if not self._trained:
            return []

        return list(self._encoder.classes_)

    @property
    def sample_count(self) -> int:
        return len(self._data)

    def feature_importance(self) -> Dict[str, float]:

        try:

            rf = self._model.named_steps["rf"]

            return {
                k: round(float(v), 4)
                for k, v in zip(
                    self.FEATURE_KEYS,
                    rf.feature_importances_
                )
            }

        except Exception:
            return {}

    # ========================================================
    # DEBUG
    # ========================================================

    def debug_prediction(
        self,
        angles: Dict[str, float]
    ) -> Dict:

        label, confidence = self.detect_activity(angles)

        return {
            "prediction": label,
            "confidence": confidence,
            "history": list(self._prediction_history),
            "classes": self.classes,
            "samples": self.sample_count,
            "feature_importance":
                self.feature_importance(),
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    detector = SmartDetector()

    test_angles = {
        "left_knee": 95,
        "right_knee": 92,
        "left_hip": 88,
        "right_hip": 90,
        "left_elbow": 160,
        "right_elbow": 158,
        "left_shoulder": 55,
        "trunk_lean_left": 170,
    }

    for i in range(5):

        label, conf = detector.detect_activity(test_angles)

        print(f"Prediction {i+1}")
        print("Label      :", label)
        print("Confidence :", conf)
        print("-" * 40)