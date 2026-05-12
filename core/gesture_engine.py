# ============================================
# core/gesture_engine.py
# ============================================

"""
Advanced AI Gesture Engine

Features:
- MediaPipe Landmark support
- Visibility filtering
- Temporal smoothing
- Hysteresis stabilization
- Multi-gesture recognition
- Confidence scoring
- Fitness AI compatible
- Real-time safe
- Crash-safe landmark handling
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from core.pose_engine import Landmark


# ============================================
# ENUMS
# ============================================

class GestureType(str, Enum):

    HANDS_UP = "Hands Up"
    HANDS_DOWN = "Hands Down"

    LEFT_HAND_UP = "Left Hand Up"
    RIGHT_HAND_UP = "Right Hand Up"

    T_POSE = "T Pose"

    FLEX = "Flex"

    PRAYER = "Prayer"

    CLAP = "Clap"

    CROSS_ARMS = "Cross Arms"

    HANDS_ON_HEAD = "Hands On Head"

    WAVE_LEFT = "Wave Left"
    WAVE_RIGHT = "Wave Right"

    STOP = "Stop"

    PUNCH_LEFT = "Punch Left"
    PUNCH_RIGHT = "Punch Right"

    NEUTRAL = "Neutral"

    UNKNOWN = "Unknown"


# ============================================
# RESULT
# ============================================

@dataclass
class GestureResult:

    gesture: GestureType

    raw_gesture: GestureType

    confidence: float

    stable: bool

    message: str


# ============================================
# ENGINE
# ============================================

class GestureEngine:

    VISIBILITY_THRESHOLD = 0.5

    def __init__(
        self,
        window_size: int = 5,
        threshold: float = 0.03,
    ):

        self.window_size = window_size

        self.threshold = threshold

        self.gesture_history = deque(
            maxlen=window_size
        )

        self._last_stable = (
            GestureType.UNKNOWN
        )

        # Wave tracking
        self._prev_left_wrist_x = None
        self._prev_right_wrist_x = None

    # ============================================================ #

    def detect_gesture(
        self,
        landmarks: List[Optional[Landmark]]
    ) -> GestureResult:

        # ------------------------------------------------ #
        # Safe landmark validation
        # ------------------------------------------------ #

        valid = self._validate_landmarks(
            landmarks
        )

        if not valid:

            return GestureResult(

                gesture=GestureType.UNKNOWN,

                raw_gesture=GestureType.UNKNOWN,

                confidence=0.0,

                stable=False,

                message="Pose landmarks missing."
            )

        # ------------------------------------------------ #

        raw_gesture = self._classify(
            landmarks
        )

        self.gesture_history.append(
            raw_gesture
        )

        # ------------------------------------------------ #
        # Majority Vote Smoothing
        # ------------------------------------------------ #

        counts = {}

        for g in self.gesture_history:
            counts[g] = counts.get(g, 0) + 1

        stable_gesture = max(
            counts,
            key=counts.get
        )

        confidence = round(
            counts[stable_gesture]
            / len(self.gesture_history),
            2
        )

        is_stable = confidence >= 0.6

        # ------------------------------------------------ #
        # Hysteresis
        # ------------------------------------------------ #

        if is_stable:
            self._last_stable = stable_gesture
        else:
            stable_gesture = self._last_stable

        # ------------------------------------------------ #

        return GestureResult(

            gesture=stable_gesture,

            raw_gesture=raw_gesture,

            confidence=confidence,

            stable=is_stable,

            message=self._message(
                stable_gesture
            )
        )

    # ============================================================ #

    def _message(
        self,
        gesture: GestureType
    ) -> str:

        messages = {

            GestureType.HANDS_UP:
                "Both hands raised.",

            GestureType.HANDS_DOWN:
                "Both hands lowered.",

            GestureType.LEFT_HAND_UP:
                "Left hand raised.",

            GestureType.RIGHT_HAND_UP:
                "Right hand raised.",

            GestureType.T_POSE:
                "T-pose detected.",

            GestureType.FLEX:
                "Flex pose detected.",

            GestureType.PRAYER:
                "Prayer pose detected.",

            GestureType.CLAP:
                "Clap gesture detected.",

            GestureType.CROSS_ARMS:
                "Arms crossed.",

            GestureType.HANDS_ON_HEAD:
                "Hands on head detected.",

            GestureType.WAVE_LEFT:
                "Left hand waving.",

            GestureType.WAVE_RIGHT:
                "Right hand waving.",

            GestureType.STOP:
                "Stop gesture detected.",

            GestureType.PUNCH_LEFT:
                "Left punch detected.",

            GestureType.PUNCH_RIGHT:
                "Right punch detected.",

            GestureType.NEUTRAL:
                "Neutral position.",

            GestureType.UNKNOWN:
                "Gesture not recognized.",
        }

        return messages.get(
            gesture,
            "Unknown gesture."
        )

    # ============================================================ #

    def _validate_landmarks(
        self,
        landmarks
    ):

        # ------------------------------------------------ #
        # Type check
        # ------------------------------------------------ #

        if not isinstance(landmarks, list):
            return False

        # ------------------------------------------------ #
        # Required upper body landmarks
        # ------------------------------------------------ #

        required = [
            11, 12,
            13, 14,
            15, 16
        ]

        # ------------------------------------------------ #
        # Safe validation
        # ------------------------------------------------ #

        for idx in required:

            if idx >= len(landmarks):
                return False

            lm = landmarks[idx]

            if lm is None:
                return False

            if not isinstance(lm, Landmark):
                return False

            if lm.visibility < self.VISIBILITY_THRESHOLD:
                return False

        return True

    # ============================================================ #

    def _visible(
        self,
        lm: Landmark
    ) -> bool:

        return (
            lm.visibility >=
            self.VISIBILITY_THRESHOLD
        )

    # ============================================================ #

    def _dist(
        self,
        a: Landmark,
        b: Landmark
    ) -> float:

        dx = a.x - b.x
        dy = a.y - b.y

        return (
            dx * dx + dy * dy
        ) ** 0.5

    # ============================================================ #

    def _classify(
        self,
        lm
    ) -> GestureType:

        try:

            ls = lm[11]
            rs = lm[12]

            le = lm[13]
            re = lm[14]

            lw = lm[15]
            rw = lm[16]

            # ------------------------------------------------ #
            # Basic arm positions
            # ------------------------------------------------ #

            left_diff = ls.y - lw.y
            right_diff = rs.y - rw.y

            left_up = left_diff > self.threshold
            right_up = right_diff > self.threshold

            left_down = left_diff < -self.threshold
            right_down = right_diff < -self.threshold

            # ------------------------------------------------ #
            # HANDS UP
            # ------------------------------------------------ #

            if left_up and right_up:
                return GestureType.HANDS_UP

            # ------------------------------------------------ #
            # HANDS DOWN
            # ------------------------------------------------ #

            if left_down and right_down:
                return GestureType.HANDS_DOWN

            # ------------------------------------------------ #
            # LEFT / RIGHT UP
            # ------------------------------------------------ #

            if left_up and not right_up:
                return GestureType.LEFT_HAND_UP

            if right_up and not left_up:
                return GestureType.RIGHT_HAND_UP

            # ------------------------------------------------ #
            # T POSE
            # ------------------------------------------------ #

            if (
                abs(lw.y - ls.y) < 0.05 and
                abs(rw.y - rs.y) < 0.05
            ):
                return GestureType.T_POSE

            # ------------------------------------------------ #
            # FLEX
            # ------------------------------------------------ #

            if (
                self._dist(lw, ls) < 0.15 and
                self._dist(rw, rs) < 0.15
            ):
                return GestureType.FLEX

            # ------------------------------------------------ #
            # PRAYER
            # ------------------------------------------------ #

            if self._dist(lw, rw) < 0.08:
                return GestureType.PRAYER

            # ------------------------------------------------ #
            # CLAP
            # ------------------------------------------------ #

            if (
                self._dist(lw, rw) < 0.04 and
                abs(lw.y - rw.y) < 0.05
            ):
                return GestureType.CLAP

            # ------------------------------------------------ #
            # CROSS ARMS
            # ------------------------------------------------ #

            if (
                lw.x > rs.x and
                rw.x < ls.x
            ):
                return GestureType.CROSS_ARMS

            # ------------------------------------------------ #
            # HANDS ON HEAD
            # ------------------------------------------------ #

            if (
                lw.y < ls.y and
                rw.y < rs.y and
                abs(lw.x - ls.x) < 0.12 and
                abs(rw.x - rs.x) < 0.12
            ):
                return GestureType.HANDS_ON_HEAD

            # ------------------------------------------------ #
            # STOP
            # ------------------------------------------------ #

            if (
                left_up and
                abs(lw.x - ls.x) > 0.15
            ):
                return GestureType.STOP

            # ------------------------------------------------ #
            # PUNCH
            # ------------------------------------------------ #

            if (
                abs(lw.x - ls.x) > 0.30 and
                abs(lw.y - ls.y) < 0.10
            ):
                return GestureType.PUNCH_LEFT

            if (
                abs(rw.x - rs.x) > 0.30 and
                abs(rw.y - rs.y) < 0.10
            ):
                return GestureType.PUNCH_RIGHT

            # ------------------------------------------------ #
            # WAVE LEFT
            # ------------------------------------------------ #

            if self._prev_left_wrist_x is not None:

                dx = abs(
                    lw.x -
                    self._prev_left_wrist_x
                )

                if dx > 0.08 and left_up:
                    self._prev_left_wrist_x = lw.x
                    return GestureType.WAVE_LEFT

            # ------------------------------------------------ #
            # WAVE RIGHT
            # ------------------------------------------------ #

            if self._prev_right_wrist_x is not None:

                dx = abs(
                    rw.x -
                    self._prev_right_wrist_x
                )

                if dx > 0.08 and right_up:
                    self._prev_right_wrist_x = rw.x
                    return GestureType.WAVE_RIGHT

            # ------------------------------------------------ #
            # Update tracking
            # ------------------------------------------------ #

            self._prev_left_wrist_x = lw.x
            self._prev_right_wrist_x = rw.x

            return GestureType.NEUTRAL

        except Exception:
            return GestureType.UNKNOWN

    # ============================================================ #

    def reset(self):

        self.gesture_history.clear()

        self._last_stable = (
            GestureType.UNKNOWN
        )

        self._prev_left_wrist_x = None
        self._prev_right_wrist_x = None

    # ============================================================ #

    def get_history(self):

        return list(
            self.gesture_history
        )