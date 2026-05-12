"""
motion_analysis.py
------------------
Advanced AI motion analysis engine.

Features:
- Rep counting
- ROM tracking
- Velocity estimation
- Exercise classification
- Phase detection
- Symmetry analysis
- Confidence scoring
- Temporal smoothing
"""

import numpy as np

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from core.landmark_extractor import BodySnapshot


# ===================================================================== #
# ENUMS
# ===================================================================== #

class MovementPhase(str, Enum):
    CONCENTRIC = "concentric"
    ECCENTRIC = "eccentric"
    REST = "rest"
    UNKNOWN = "unknown"


class MovementPattern(str, Enum):
    SQUAT = "Squat"
    DEADLIFT = "Deadlift"
    PUSH_UP = "Push-up"
    CURL = "Curl"
    PRESS = "Press"
    LUNGE = "Lunge"
    PLANK = "Plank"
    JUMPING_JACK = "Jumping Jack"
    JUMP = "Jump"
    UNKNOWN = "Unknown"


# ===================================================================== #
# DATA CLASSES
# ===================================================================== #

@dataclass
class RepEvent:
    rep_number: int
    angle_key: str
    min_angle: float
    max_angle: float
    duration_frames: int
    phase: MovementPhase = MovementPhase.UNKNOWN


@dataclass
class ROMSummary:
    angle_key: str
    current: float
    minimum: float
    maximum: float
    range: float


@dataclass
class MotionState:

    pattern: MovementPattern = MovementPattern.UNKNOWN

    phase: MovementPhase = MovementPhase.UNKNOWN

    rep_count: int = 0

    roms: Dict[str, ROMSummary] = field(default_factory=dict)

    recent_reps: List[RepEvent] = field(default_factory=list)

    velocity: Dict[str, float] = field(default_factory=dict)

    is_moving: bool = False

    confidence: float = 0.0

    symmetry_score: Optional[float] = None  # FIX 3: None means data missing, 0.0 means perfectly symmetric


# ===================================================================== #
# REP COUNTER
# ===================================================================== #

class RepCounter:

    def __init__(
        self,
        angle_key: str,
        low_threshold: float,
        high_threshold: float,
        direction: str = "down_then_up",
        min_frames_in_phase: int = 3,
    ):

        self.angle_key = angle_key

        self.low_th = low_threshold
        self.high_th = high_threshold

        self.direction = direction

        self.min_frames = min_frames_in_phase

        self._count = 0

        self._state = "waiting"

        self._phase_frames = 0

        # FIX 1: Use float('inf') instead of hardcoded 999.0
        self._min_seen = float('inf')
        self._max_seen = 0.0

        self.events: List[RepEvent] = []

    @property
    def count(self):
        return self._count

    def update(self, angle: float) -> Optional[RepEvent]:

        if angle < 0:
            return None

        self._phase_frames += 1

        self._min_seen = min(self._min_seen, angle)
        self._max_seen = max(self._max_seen, angle)

        event = None

        # ------------------------------------------------------------ #
        # DOWN → UP
        # ------------------------------------------------------------ #

        if self.direction == "down_then_up":

            if self._state == "waiting" and angle < self.low_th:

                self._state = "down"

                self._phase_frames = 0

                self._min_seen = angle

            elif self._state == "down" and angle > self.high_th:

                if self._phase_frames >= self.min_frames:

                    self._count += 1

                    event = RepEvent(
                        rep_number=self._count,
                        angle_key=self.angle_key,
                        min_angle=self._min_seen,
                        max_angle=self._max_seen,
                        duration_frames=self._phase_frames,
                        phase=MovementPhase.CONCENTRIC,
                    )

                    self.events.append(event)

                self._state = "up"

                self._phase_frames = 0

                # FIX 5: Properly reset _max_seen when transitioning to "up"
                self._max_seen = angle

            elif self._state == "up" and angle < self.low_th:

                self._state = "down"

                self._phase_frames = 0

                # FIX 5: Properly reset _min_seen when transitioning to "down"
                self._min_seen = angle

        # ------------------------------------------------------------ #
        # UP → DOWN
        # ------------------------------------------------------------ #

        elif self.direction == "up_then_down":

            if self._state == "waiting" and angle > self.high_th:

                self._state = "up"

                self._phase_frames = 0

                # FIX 5: Reset _max_seen on entering "up" state
                self._max_seen = angle

            elif self._state == "up" and angle < self.low_th:

                if self._phase_frames >= self.min_frames:

                    self._count += 1

                    event = RepEvent(
                        rep_number=self._count,
                        angle_key=self.angle_key,
                        min_angle=self._min_seen,
                        max_angle=self._max_seen,
                        duration_frames=self._phase_frames,
                        phase=MovementPhase.ECCENTRIC,
                    )

                    self.events.append(event)

                self._state = "down"

                self._phase_frames = 0

                # FIX 5: Reset _min_seen when transitioning to "down"
                self._min_seen = angle

        return event

    def reset(self):

        self._count = 0

        self._state = "waiting"

        self._phase_frames = 0

        # FIX 1: Use float('inf') on reset too
        self._min_seen = float('inf')

        self._max_seen = 0.0

        self.events.clear()


# ===================================================================== #
# ROM TRACKER
# ===================================================================== #

class ROMTracker:

    def __init__(self):

        self._min: Dict[str, float] = {}

        self._max: Dict[str, float] = {}

    def update(self, angles: Dict[str, float]):

        summaries = {}

        for key, val in angles.items():

            if val < 0:
                continue

            self._min[key] = min(
                self._min.get(key, val),
                val
            )

            self._max[key] = max(
                self._max.get(key, 0.0),
                val
            )

            summaries[key] = ROMSummary(
                angle_key=key,
                current=round(val, 1),
                minimum=round(self._min[key], 1),
                maximum=round(self._max[key], 1),
                range=round(
                    self._max[key] - self._min[key],
                    1
                ),
            )

        return summaries

    def reset(self):

        self._min.clear()
        self._max.clear()


# ===================================================================== #
# MOVEMENT CLASSIFIER
# ===================================================================== #

class MovementClassifier:

    def classify(
        self,
        angles: Dict[str, float],
        velocity: Dict[str, float],
    ) -> MovementPattern:

        lk = angles.get("left_knee", -1)
        rk = angles.get("right_knee", -1)

        lh = angles.get("left_hip", -1)
        rh = angles.get("right_hip", -1)

        le = angles.get("left_elbow", -1)
        re = angles.get("right_elbow", -1)

        ls = angles.get("left_shoulder", -1)
        rs = angles.get("right_shoulder", -1)

        trunk = angles.get("trunk_lean_avg", -1)

        knee_avg = self._avg(lk, rk)
        hip_avg = self._avg(lh, rh)

        elbow_avg = self._avg(le, re)

        shoulder_avg = self._avg(ls, rs)

        # ------------------------------------------------------------ #
        # Squat
        # ------------------------------------------------------------ #

        if 0 < knee_avg < 130 and 0 < hip_avg < 120:
            return MovementPattern.SQUAT

        # ------------------------------------------------------------ #
        # Lunge
        # ------------------------------------------------------------ #

        if lk > 0 and rk > 0 and abs(lk - rk) > 30:
            return MovementPattern.LUNGE

        # ------------------------------------------------------------ #
        # Push-up
        # ------------------------------------------------------------ #

        if (
            40 < elbow_avg < 120 and
            70 < trunk < 120 and
            abs(ls - rs) < 25
        ):
            return MovementPattern.PUSH_UP

        # ------------------------------------------------------------ #
        # Curl
        # ------------------------------------------------------------ #

        if 0 < elbow_avg < 100 and 0 < shoulder_avg < 60:
            return MovementPattern.CURL

        # ------------------------------------------------------------ #
        # Press
        # ------------------------------------------------------------ #

        if shoulder_avg > 130 and elbow_avg > 140:
            return MovementPattern.PRESS

        # ------------------------------------------------------------ #
        # Plank
        # ------------------------------------------------------------ #

        if 80 < elbow_avg < 110 and trunk < 110:
            return MovementPattern.PLANK

        # ------------------------------------------------------------ #
        # Deadlift
        # ------------------------------------------------------------ #

        if 0 < hip_avg < 100 and knee_avg > 140:
            return MovementPattern.DEADLIFT

        # ------------------------------------------------------------ #
        # Jump
        # ------------------------------------------------------------ #

        ankle_vel = velocity.get("left_ankle", 0)

        if ankle_vel > 15 and knee_avg > 150:
            return MovementPattern.JUMP

        return MovementPattern.UNKNOWN

    @staticmethod
    def _avg(*vals):

        valid = [v for v in vals if v >= 0]

        return float(np.mean(valid)) if valid else -1.0


# ===================================================================== #
# PHASE DETECTOR
# ===================================================================== #

class PhaseDetector:

    VELOCITY_THRESHOLD = 2.0

    def __init__(self, primary_key: str = "left_knee"):

        self.primary_key = primary_key

        self._vel_history = deque(maxlen=5)

    def detect(
        self,
        velocity: Dict[str, float],
        pattern: MovementPattern,
    ) -> MovementPhase:

        v = velocity.get(self.primary_key, 0.0)

        self._vel_history.append(v)

        avg_vel = float(np.mean(self._vel_history))

        if abs(avg_vel) < self.VELOCITY_THRESHOLD:
            return MovementPhase.REST

        if pattern in (
            MovementPattern.SQUAT,
            MovementPattern.CURL,
            MovementPattern.PUSH_UP,
        ):

            return (
                MovementPhase.CONCENTRIC
                if avg_vel < 0
                else MovementPhase.ECCENTRIC
            )

        return (
            MovementPhase.CONCENTRIC
            if avg_vel > 0
            else MovementPhase.ECCENTRIC
        )


# ===================================================================== #
# MOTION ANALYSER
# ===================================================================== #

class MotionAnalyser:

    EXERCISE_PRESETS = {

        "squat": {
            "angle_key": "left_knee",
            "low_threshold": 100,
            "high_threshold": 160,
        },

        "curl": {
            "angle_key": "left_elbow",
            "low_threshold": 60,
            "high_threshold": 150,
        },

        "push_up": {
            "angle_key": "left_elbow",
            "low_threshold": 70,
            "high_threshold": 160,
            "direction": "down_then_up",
        },

        "lunge": {
            "angle_key": "left_knee",
            "low_threshold": 100,
            "high_threshold": 165,
        },

        "deadlift": {
            "angle_key": "left_hip",
            "low_threshold": 70,
            "high_threshold": 160,
        },
    }

    def __init__(self, exercise: str = "auto"):

        self._exercise = exercise

        self._rep_counter: Optional[RepCounter] = None

        self._rom = ROMTracker()

        self._classifier = MovementClassifier()

        self._phase_det = PhaseDetector()

        self._vel_history: Dict[str, deque] = {}

        self._pattern_history = deque(maxlen=15)

        self._frame = 0

        # FIX 4: Track current pattern to avoid resetting counter mid-exercise
        self._current_pattern: Optional[MovementPattern] = None

        if (
            exercise != "auto" and
            exercise in self.EXERCISE_PRESETS
        ):
            self._init_counter(exercise)

    # ---------------------------------------------------------------- #

    def _init_counter(self, exercise: str):

        p = self.EXERCISE_PRESETS[exercise]

        self._rep_counter = RepCounter(**p)

    # ---------------------------------------------------------------- #

    def update(self, snap: BodySnapshot) -> MotionState:

        self._frame += 1

        angles = snap.angles

        # ------------------------------------------------------------ #
        # Velocity
        # ------------------------------------------------------------ #

        velocity = self._compute_velocity(angles)

        # ------------------------------------------------------------ #
        # ROM
        # ------------------------------------------------------------ #

        roms = self._rom.update(angles)

        # ------------------------------------------------------------ #
        # Classification
        # ------------------------------------------------------------ #

        pattern = self._classifier.classify(
            angles,
            velocity
        )

        self._pattern_history.append(pattern)

        pattern = max(
            set(self._pattern_history),
            key=self._pattern_history.count
        )

        # ------------------------------------------------------------ #
        # Auto Exercise Detection
        # ------------------------------------------------------------ #

        if self._exercise == "auto":

            # FIX 4: Only re-init counter if pattern actually changed,
            # not blindly every 30 frames — prevents mid-exercise rep loss
            if pattern != MovementPattern.UNKNOWN and pattern != self._current_pattern:
                self._auto_init(pattern)
                self._current_pattern = pattern

        # ------------------------------------------------------------ #
        # Rep Counting
        # ------------------------------------------------------------ #

        if self._rep_counter:

            angle_val = angles.get(
                self._rep_counter.angle_key,
                -1
            )

            self._rep_counter.update(angle_val)

        # ------------------------------------------------------------ #
        # Phase
        # ------------------------------------------------------------ #

        phase = self._phase_det.detect(
            velocity,
            pattern
        )

        # ------------------------------------------------------------ #
        # Movement
        # ------------------------------------------------------------ #

        any_moving = any(
            abs(v) > 1.5
            for v in velocity.values()
        )

        # ------------------------------------------------------------ #
        # FIX 2: Confidence based on landmark visibility (via angles
        # available) rather than velocity alone — a slow but correct
        # rep should still have high confidence.
        # ------------------------------------------------------------ #

        total_keys = len(angles)
        valid_keys = sum(1 for v in angles.values() if v >= 0)

        confidence = round(valid_keys / total_keys, 2) if total_keys > 0 else 0.0

        # ------------------------------------------------------------ #
        # FIX 3: Symmetry — None when data is missing, otherwise the
        # actual angular difference (0.0 = perfectly symmetric).
        # ------------------------------------------------------------ #

        lk = angles.get("left_knee", -1)
        rk = angles.get("right_knee", -1)

        symmetry: Optional[float] = None

        if lk > 0 and rk > 0:
            symmetry = round(abs(lk - rk), 2)

        return MotionState(

            pattern=pattern,

            phase=phase,

            rep_count=(
                self._rep_counter.count
                if self._rep_counter else 0
            ),

            roms=roms,

            recent_reps=(
                self._rep_counter.events[-5:]
                if self._rep_counter else []
            ),

            velocity=velocity,

            is_moving=any_moving,

            confidence=confidence,

            symmetry_score=symmetry,
        )

    # ---------------------------------------------------------------- #

    def reset_reps(self):

        if self._rep_counter:
            self._rep_counter.reset()

    # ---------------------------------------------------------------- #

    def _auto_init(self, pattern: MovementPattern):

        mapping = {

            MovementPattern.SQUAT: "squat",

            MovementPattern.CURL: "curl",

            MovementPattern.PUSH_UP: "push_up",

            MovementPattern.LUNGE: "lunge",

            MovementPattern.DEADLIFT: "deadlift",
        }

        key = mapping.get(pattern)

        if key:
            self._init_counter(key)

    # ---------------------------------------------------------------- #

    def _compute_velocity(
        self,
        angles: Dict[str, float]
    ) -> Dict[str, float]:

        velocity = {}

        for key, val in angles.items():

            if val < 0:
                continue

            if key not in self._vel_history:

                self._vel_history[key] = deque(maxlen=5)

            buf = self._vel_history[key]

            buf.append(val)

            if len(buf) >= 2:

                diffs = np.diff(buf)

                velocity[key] = float(
                    np.mean(diffs)
                )

            else:
                velocity[key] = 0.0

        return velocity