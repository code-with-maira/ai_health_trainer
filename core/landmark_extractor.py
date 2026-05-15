"""
landmark_extractor.py
Ultra Advanced Landmark Extraction
"""

import numpy as np

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from collections import deque

from core.pose_engine import (
    PoseResult,
    Landmark,
    PoseEngine,
)

# ============================================================
# JOINT IDS
# ============================================================

class J:

    NOSE = 0

    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14

    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    LEFT_HIP = 23
    RIGHT_HIP = 24

    LEFT_KNEE = 25
    RIGHT_KNEE = 26

    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


# ============================================================
# BODY SNAPSHOT
# ============================================================

@dataclass
class BodySnapshot:

    nose: Optional[Landmark] = None

    left_shoulder: Optional[Landmark] = None
    right_shoulder: Optional[Landmark] = None

    left_elbow: Optional[Landmark] = None
    right_elbow: Optional[Landmark] = None

    left_wrist: Optional[Landmark] = None
    right_wrist: Optional[Landmark] = None

    left_hip: Optional[Landmark] = None
    right_hip: Optional[Landmark] = None

    left_knee: Optional[Landmark] = None
    right_knee: Optional[Landmark] = None

    left_ankle: Optional[Landmark] = None
    right_ankle: Optional[Landmark] = None

    angles: Dict[str, float] = field(default_factory=dict)

    velocities: Dict[str, float] = field(default_factory=dict)

    accelerations: Dict[str, float] = field(default_factory=dict)

    fatigue: Dict[str, float] = field(default_factory=dict)

    posture: Dict[str, float] = field(default_factory=dict)

    symmetry: Dict[str, float] = field(default_factory=dict)

    distances: Dict[str, float] = field(default_factory=dict)

    fully_visible: bool = False

    # ========================================================
    # IMPORTANT FIXES
    # ========================================================

    def to_list(self):

        joints = [

            self.nose,

            self.left_shoulder,
            self.right_shoulder,

            self.left_elbow,
            self.right_elbow,

            self.left_wrist,
            self.right_wrist,

            self.left_hip,
            self.right_hip,

            self.left_knee,
            self.right_knee,

            self.left_ankle,
            self.right_ankle,
        ]

        output = []

        for lm in joints:

            if lm is None:

                output.extend([0.0, 0.0])

            else:

                output.extend([lm.x, lm.y])

        return output

    def to_numpy(self):

        return np.array(
            self.to_list(),
            dtype=np.float32
        )

    def __iter__(self):

        return iter(self.to_list())

    def __len__(self):

        return len(self.to_list())


# ============================================================
# LANDMARK EXTRACTOR
# ============================================================

class LandmarkExtractor:

    VIS_THRESHOLD = 0.5

    def __init__(self):

        self.history = deque(maxlen=10)

        self.prev_angles = {}

        self.prev_velocity = {}

    # ========================================================

    def reset(self):

        self.history.clear()

        self.prev_angles.clear()

        self.prev_velocity.clear()

    # ========================================================

    def extract(
        self,
        result: PoseResult
    ) -> Optional[BodySnapshot]:

        if (
            not result.pose_present or
            not result.landmarks
        ):

            self.reset()

            return None

        snap = BodySnapshot()

        # ====================================================
        # GET LANDMARK
        # ====================================================

        def get(idx):

            lm = result.get(idx)

            if (
                lm and
                lm.visibility >= self.VIS_THRESHOLD
            ):

                return lm

            return None

        # ====================================================
        # ASSIGN
        # ====================================================

        snap.nose = get(J.NOSE)

        snap.left_shoulder = get(J.LEFT_SHOULDER)
        snap.right_shoulder = get(J.RIGHT_SHOULDER)

        snap.left_elbow = get(J.LEFT_ELBOW)
        snap.right_elbow = get(J.RIGHT_ELBOW)

        snap.left_wrist = get(J.LEFT_WRIST)
        snap.right_wrist = get(J.RIGHT_WRIST)

        snap.left_hip = get(J.LEFT_HIP)
        snap.right_hip = get(J.RIGHT_HIP)

        snap.left_knee = get(J.LEFT_KNEE)
        snap.right_knee = get(J.RIGHT_KNEE)

        snap.left_ankle = get(J.LEFT_ANKLE)
        snap.right_ankle = get(J.RIGHT_ANKLE)

        # ====================================================
        # ANGLES
        # ====================================================

        a = snap.angles

        a["left_elbow"] = self._angle(
            snap.left_shoulder,
            snap.left_elbow,
            snap.left_wrist
        )

        a["right_elbow"] = self._angle(
            snap.right_shoulder,
            snap.right_elbow,
            snap.right_wrist
        )

        a["left_knee"] = self._angle(
            snap.left_hip,
            snap.left_knee,
            snap.left_ankle
        )

        a["right_knee"] = self._angle(
            snap.right_hip,
            snap.right_knee,
            snap.right_ankle
        )

        # ====================================================
        # VELOCITY
        # ====================================================

        for key, val in a.items():

            prev = self.prev_angles.get(key)

            if prev is None:

                velocity = 0.0

            else:

                velocity = val - prev

            snap.velocities[key] = velocity

            self.prev_angles[key] = val

        # ====================================================
        # FATIGUE
        # ====================================================

        if snap.velocities:

            fatigue_score = np.mean(

                np.abs(
                    list(snap.velocities.values())
                )

            )

        else:

            fatigue_score = 0.0

        snap.fatigue["score"] = float(
            fatigue_score
        )

        # ====================================================
        # VISIBILITY
        # ====================================================

        required = [

            snap.left_shoulder,
            snap.right_shoulder,

            snap.left_hip,
            snap.right_hip,
        ]

        snap.fully_visible = all(
            x is not None
            for x in required
        )

        self.history.append(snap)

        return snap

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _angle(a, b, c):

        if (
            a is None or
            b is None or
            c is None
        ):

            return 0.0

        return PoseEngine.landmark_angle(
            a,
            b,
            c
        )

    # ========================================================
    # FEATURE VECTOR
    # ========================================================

    def extract_angle_vector(
        self,
        snap: BodySnapshot
    ):

        keys = [

            "left_elbow",
            "right_elbow",

            "left_knee",
            "right_knee",
        ]

        return np.array(

            [

                snap.angles.get(k, 0.0)

                for k in keys

            ],

            dtype=np.float32
        )