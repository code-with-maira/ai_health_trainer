# ============================================
# core/injury_detection.py
# ============================================

import math

from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ============================================
# DATA CLASS
# ============================================

@dataclass
class InjuryResult:

    risk_level: str

    confidence: float

    message: str

    recommendations: List[str] = field(default_factory=list)

    knee_distance: float = 0.0

    knee_symmetry_diff: float = 0.0

    hip_symmetry_diff: float = 0.0

    left_knee_angle: float = 0.0
    right_knee_angle: float = 0.0

    stability_score: float = 0.0

    flags: List[str] = field(default_factory=list)


# ============================================
# INJURY DETECTOR
# ============================================

class InjuryDetection:

    LANDMARKS = {

        "left_hip": 23,
        "right_hip": 24,

        "left_knee": 25,
        "right_knee": 26,

        "left_ankle": 27,
        "right_ankle": 28,
    }

    # ------------------------------------------------------------ #

    def __init__(
        self,
        window_size: int = 10,
    ):

        self.window_size = window_size

        self.distance_history = deque(maxlen=window_size)

        self.symmetry_history = deque(maxlen=window_size)

        self.angle_history = deque(maxlen=window_size)

    # ============================================================ #
    # VALIDATION
    # ============================================================ #

    def _validate(self, landmarks: List[Dict]):

        if not isinstance(landmarks, list):
            raise TypeError("landmarks must be a list")

        required = self.LANDMARKS.values()

        for idx in required:

            if idx >= len(landmarks):
                raise IndexError(f"Missing landmark index {idx}")

            lm = landmarks[idx]

            if not isinstance(lm, dict):
                raise TypeError(f"Landmark {idx} must be dict")

            for key in ("x", "y"):

                if key not in lm:
                    raise KeyError(f"Landmark {idx} missing '{key}'")

                if not isinstance(lm[key], (int, float)):
                    raise TypeError(
                        f"Landmark {idx}['{key}'] must be number"
                    )

    # ============================================================ #
    # HELPERS
    # ============================================================ #

    @staticmethod
    def _point(lm: Dict):

        return (lm["x"], lm["y"])

    # ------------------------------------------------------------ #

    @staticmethod
    def distance(a, b):

        return math.sqrt(
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2
        )

    # ------------------------------------------------------------ #

    @staticmethod
    def angle(a, b, c):

        ba = (a[0] - b[0], a[1] - b[1])

        bc = (c[0] - b[0], c[1] - b[1])

        dot = ba[0] * bc[0] + ba[1] * bc[1]

        mag1 = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        mag2 = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

        if mag1 * mag2 == 0:
            return 0.0

        cos_angle = dot / (mag1 * mag2)

        cos_angle = max(-1.0, min(1.0, cos_angle))

        return round(
            math.degrees(math.acos(cos_angle)),
            2
        )

    # ============================================================ #
    # MAIN DETECTION
    # ============================================================ #

    def detect_risk(
        self,
        landmarks: List[Dict],
    ) -> InjuryResult:

        self._validate(landmarks)

        # -------------------------------------------------------- #
        # POINTS
        # -------------------------------------------------------- #

        lh = self._point(landmarks[23])
        rh = self._point(landmarks[24])

        lk = self._point(landmarks[25])
        rk = self._point(landmarks[26])

        la = self._point(landmarks[27])
        ra = self._point(landmarks[28])

        # -------------------------------------------------------- #
        # DISTANCES
        # -------------------------------------------------------- #

        knee_distance = round(
            self.distance(lk, rk),
            4
        )

        self.distance_history.append(knee_distance)

        avg_distance = (
            sum(self.distance_history)
            / len(self.distance_history)
        )

        # -------------------------------------------------------- #
        # SYMMETRY
        # -------------------------------------------------------- #

        knee_symmetry = abs(lk[1] - rk[1])

        hip_symmetry = abs(lh[1] - rh[1])

        self.symmetry_history.append(knee_symmetry)

        # -------------------------------------------------------- #
        # ANGLES
        # -------------------------------------------------------- #

        left_knee_angle = self.angle(
            lh, lk, la
        )

        right_knee_angle = self.angle(
            rh, rk, ra
        )

        avg_angle = (
            left_knee_angle +
            right_knee_angle
        ) / 2

        self.angle_history.append(avg_angle)

        # -------------------------------------------------------- #
        # STABILITY SCORE
        # -------------------------------------------------------- #

        stability_score = max(
            0.0,
            100 - (
                knee_symmetry * 1000
            )
        )

        stability_score = round(stability_score, 1)

        # -------------------------------------------------------- #
        # FLAGS
        # -------------------------------------------------------- #

        flags = []

        # Knee valgus
        if avg_distance < 0.04:
            flags.append("knee_valgus")

        # Knee varus
        if avg_distance > 0.16:
            flags.append("wide_stance")

        # Deep bend
        if avg_angle < 70:
            flags.append("deep_knee_flexion")

        # Poor symmetry
        if knee_symmetry > 0.03:
            flags.append("asymmetry")

        # Hip imbalance
        if hip_symmetry > 0.03:
            flags.append("hip_instability")

        # -------------------------------------------------------- #
        # RISK LEVEL
        # -------------------------------------------------------- #

        if "knee_valgus" in flags:

            risk = "High Risk"

            message = (
                "Knees collapsing inward detected."
            )

            recommendations = [
                "Stop heavy exercise",
                "Strengthen glutes",
                "Check squat form",
                "Reduce training load",
            ]

        elif "asymmetry" in flags:

            risk = "Medium Risk"

            message = (
                "Movement asymmetry detected."
            )

            recommendations = [
                "Focus on balance",
                "Use lighter weight",
                "Monitor left/right control",
            ]

        elif "wide_stance" in flags:

            risk = "Warning"

            message = (
                "Very wide stance detected."
            )

            recommendations = [
                "Adjust stance width",
                "Check ankle mobility",
            ]

        else:

            risk = "Safe"

            message = (
                "Movement pattern looks stable."
            )

            recommendations = [
                "Continue training",
                "Maintain proper warm-up",
            ]

        # -------------------------------------------------------- #
        # CONFIDENCE
        # -------------------------------------------------------- #

        confidence = round(
            min(
                1.0,
                len(self.distance_history)
                / self.window_size
            ),
            2
        )

        # -------------------------------------------------------- #
        # RETURN
        # -------------------------------------------------------- #

        return InjuryResult(

            risk_level = risk,

            confidence = confidence,

            message = message,

            recommendations = recommendations,

            knee_distance = round(knee_distance, 4),

            knee_symmetry_diff = round(knee_symmetry, 4),

            hip_symmetry_diff = round(hip_symmetry, 4),

            left_knee_angle = left_knee_angle,

            right_knee_angle = right_knee_angle,

            stability_score = stability_score,

            flags = flags,
        )

    # ============================================================ #
    # TREND
    # ============================================================ #

    def get_trend(self):

        if len(self.distance_history) < 3:
            return "Insufficient Data"

        diffs = [

            self.distance_history[i]
            - self.distance_history[i - 1]

            for i in range(
                1,
                len(self.distance_history)
            )
        ]

        avg = sum(diffs) / len(diffs)

        if avg > 0.005:
            return "Improving"

        if avg < -0.005:
            return "Worsening"

        return "Stable"

    # ============================================================ #

    def reset(self):

        self.distance_history.clear()

        self.symmetry_history.clear()

        self.angle_history.clear()


# ============================================
# TEST
# ============================================

if __name__ == "__main__":

    detector = InjuryDetection()

    landmarks = [

        {"x": 0.5, "y": 0.5}

        for _ in range(33)
    ]

    landmarks[23] = {"x": 0.45, "y": 0.40}
    landmarks[24] = {"x": 0.55, "y": 0.40}

    landmarks[25] = {"x": 0.47, "y": 0.60}
    landmarks[26] = {"x": 0.53, "y": 0.61}

    landmarks[27] = {"x": 0.46, "y": 0.85}
    landmarks[28] = {"x": 0.54, "y": 0.85}

    result = detector.detect_risk(landmarks)

    print("\n=== Injury Detection ===")

    print("Risk:", result.risk_level)

    print("Confidence:", result.confidence)

    print("Flags:", result.flags)

    print("Left Knee Angle:", result.left_knee_angle)

    print("Right Knee Angle:", result.right_knee_angle)

    print("Stability:", result.stability_score)

    print("Trend:", detector.get_trend())

    print("Message:", result.message)