# ============================================
# core/fatigue_engine.py
# ============================================

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List


# ============================================
# ENUM
# ============================================

class FatigueLevel(str, Enum):

    FRESH = "fresh"
    ACTIVE = "active"
    MODERATE = "moderate"
    TIRED = "tired"
    SEVERE = "severe"


# ============================================
# RESULT OBJECT
# ============================================

@dataclass
class FatigueResult:

    level: FatigueLevel
    confidence: float
    diff: float
    avg_diff: float
    message: str


# ============================================
# ENGINE
# ============================================

class FatigueEngine:
    """
    AI fatigue detection using elbow angle movement.
    """

    def __init__(self, window_size: int = 5):

        self.previous_angle = None

        self.window_size = window_size

        self.diff_history = deque(maxlen=window_size)

    # ============================================================ #

    def detect_fatigue(
        self,
        motion_data: dict
    ) -> FatigueResult:

        if not isinstance(motion_data, dict):
            raise TypeError(
                "motion_data must be dict"
            )

        if "elbow_angle" not in motion_data:
            raise KeyError(
                "'elbow_angle' missing"
            )

        angle = motion_data["elbow_angle"]

        if not isinstance(angle, (int, float)):
            raise TypeError(
                "elbow_angle must be int/float"
            )

        if not (0 <= angle <= 180):
            raise ValueError(
                f"Invalid elbow angle: {angle}"
            )

        # -------------------------------------------------------- #
        # First frame
        # -------------------------------------------------------- #

        if self.previous_angle is None:

            self.previous_angle = angle

            return FatigueResult(
                level       = FatigueLevel.FRESH,
                confidence  = 1.0,
                diff        = 0.0,
                avg_diff    = 0.0,
                message     = (
                    "Initial reading recorded."
                )
            )

        # -------------------------------------------------------- #

        diff = abs(
            angle - self.previous_angle
        )

        self.previous_angle = angle

        self.diff_history.append(diff)

        avg_diff = (
            sum(self.diff_history)
            / len(self.diff_history)
        )

        confidence = round(
            len(self.diff_history)
            / self.window_size,
            2
        )

        # -------------------------------------------------------- #
        # Fatigue Logic
        # -------------------------------------------------------- #

        if avg_diff >= 10:

            level = FatigueLevel.ACTIVE

            message = (
                "Athlete fully active."
            )

        elif avg_diff >= 5:

            level = FatigueLevel.MODERATE

            message = (
                "Fatigue starting to appear."
            )

        elif avg_diff >= 2:

            level = FatigueLevel.TIRED

            message = (
                "Athlete appears tired."
            )

        else:

            level = FatigueLevel.SEVERE

            message = (
                "Severe fatigue detected."
            )

        return FatigueResult(

            level       = level,

            confidence  = confidence,

            diff        = round(diff, 2),

            avg_diff    = round(avg_diff, 2),

            message     = message,
        )

    # ============================================================ #

    def reset(self):

        self.previous_angle = None

        self.diff_history.clear()

    # ============================================================ #

    def get_history(self) -> List[float]:

        return list(self.diff_history)