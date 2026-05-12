"""
landmark_extractor.py
---------------------
ULTRA Advanced Body Landmark Extraction + Biomechanics AI

Improved Features:
- Structured body snapshot
- 2D + 3D joint angles
- Distance metrics
- Body symmetry analysis
- Center of mass
- Stability estimation
- Velocity tracking
- Acceleration tracking
- Temporal smoothing
- Posture metrics
- Balance analysis
- ML-ready feature vectors
- Fatigue indicators
- Real-time optimized
"""

import numpy as np

from dataclasses import dataclass, field

from typing import Optional, Dict, Tuple, List

from collections import deque

from core.pose_engine import (
    PoseResult,
    Landmark,
    PoseEngine,
)


# ===================================================================== #
# JOINT INDICES
# ===================================================================== #

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

    LEFT_HEEL = 29
    RIGHT_HEEL = 30

    LEFT_FOOT = 31
    RIGHT_FOOT = 32


# ===================================================================== #
# BODY SNAPSHOT
# ===================================================================== #

@dataclass
class BodySnapshot:

    # ------------------------------------------------------------ #
    # Core joints
    # ------------------------------------------------------------ #

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

    # ------------------------------------------------------------ #
    # Computed metrics
    # ------------------------------------------------------------ #

    angles: Dict[str, float] = field(default_factory=dict)

    angles_3d: Dict[str, float] = field(default_factory=dict)

    distances: Dict[str, float] = field(default_factory=dict)

    symmetry: Dict[str, float] = field(default_factory=dict)

    velocities: Dict[str, float] = field(default_factory=dict)

    accelerations: Dict[str, float] = field(default_factory=dict)

    posture: Dict[str, float] = field(default_factory=dict)

    fatigue: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------ #
    # Body geometry
    # ------------------------------------------------------------ #

    mid_shoulder: Optional[Tuple[int, int]] = None

    mid_hip: Optional[Tuple[int, int]] = None

    center_of_mass: Optional[Tuple[int, int]] = None

    body_scale: float = 1.0

    stability_score: float = 0.0

    balance_score: float = 0.0

    # ------------------------------------------------------------ #

    fully_visible: bool = False


# ===================================================================== #
# EXTRACTOR
# ===================================================================== #

class LandmarkExtractor:

    VIS_THRESHOLD = 0.5

    MISSING = -1.0

    def __init__(self):

        self._prev_angles: Dict[str, float] = {}

        self._prev_velocities: Dict[str, float] = {}

        self._history = deque(maxlen=10)

    # ================================================================= #

    def reset(self):

        self._prev_angles.clear()

        self._prev_velocities.clear()

        self._history.clear()

    # ================================================================= #

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

        # ------------------------------------------------------------ #
        # Joint Getter
        # ------------------------------------------------------------ #

        def get(idx):

            lm = result.get(idx)

            return (
                lm
                if (
                    lm and
                    lm.visibility >= self.VIS_THRESHOLD
                )
                else None
            )

        # ------------------------------------------------------------ #
        # Assign Joints
        # ------------------------------------------------------------ #

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

        # ------------------------------------------------------------ #
        # Midpoints
        # ------------------------------------------------------------ #

        if (
            snap.left_shoulder and
            snap.right_shoulder
        ):

            snap.mid_shoulder = self._midpoint(
                snap.left_shoulder,
                snap.right_shoulder
            )

        if (
            snap.left_hip and
            snap.right_hip
        ):

            snap.mid_hip = self._midpoint(
                snap.left_hip,
                snap.right_hip
            )

        # ------------------------------------------------------------ #
        # Center of Mass
        # ------------------------------------------------------------ #

        visible = [

            lm for lm in [

                snap.left_shoulder,
                snap.right_shoulder,

                snap.left_hip,
                snap.right_hip,

                snap.left_knee,
                snap.right_knee,

                snap.left_ankle,
                snap.right_ankle,

            ]

            if lm is not None
        ]

        if visible:

            xs = [lm.px for lm in visible]
            ys = [lm.py for lm in visible]

            snap.center_of_mass = (

                int(np.mean(xs)),
                int(np.mean(ys))

            )

        # ------------------------------------------------------------ #
        # Angles
        # ------------------------------------------------------------ #

        a = snap.angles
        a3 = snap.angles_3d

        # Elbows

        a["left_elbow"] = self._angle3(
            snap.left_shoulder,
            snap.left_elbow,
            snap.left_wrist
        )

        a["right_elbow"] = self._angle3(
            snap.right_shoulder,
            snap.right_elbow,
            snap.right_wrist
        )

        # Knees

        a["left_knee"] = self._angle3(
            snap.left_hip,
            snap.left_knee,
            snap.left_ankle
        )

        a["right_knee"] = self._angle3(
            snap.right_hip,
            snap.right_knee,
            snap.right_ankle
        )

        # Shoulders

        a["left_shoulder"] = self._angle3(
            snap.left_elbow,
            snap.left_shoulder,
            snap.left_hip
        )

        a["right_shoulder"] = self._angle3(
            snap.right_elbow,
            snap.right_shoulder,
            snap.right_hip
        )

        # Hips

        a["left_hip"] = self._angle3(
            snap.left_shoulder,
            snap.left_hip,
            snap.left_knee
        )

        a["right_hip"] = self._angle3(
            snap.right_shoulder,
            snap.right_hip,
            snap.right_knee
        )

        # 3D Angles

        a3["left_elbow_3d"] = self._angle3_3d(
            snap.left_shoulder,
            snap.left_elbow,
            snap.left_wrist
        )

        a3["right_elbow_3d"] = self._angle3_3d(
            snap.right_shoulder,
            snap.right_elbow,
            snap.right_wrist
        )

        # ------------------------------------------------------------ #
        # Neck Tilt
        # ------------------------------------------------------------ #

        if (
            snap.nose and
            snap.left_shoulder and
            snap.right_shoulder
        ):

            a["neck_tilt"] = self._neck_tilt(
                snap.nose,
                snap.left_shoulder,
                snap.right_shoulder
            )

        # ------------------------------------------------------------ #
        # Distances
        # ------------------------------------------------------------ #

        d = snap.distances

        if (
            snap.left_shoulder and
            snap.right_shoulder
        ):

            d["shoulder_width"] = self._dist(
                snap.left_shoulder,
                snap.right_shoulder
            )

        if (
            snap.left_hip and
            snap.right_hip
        ):

            d["hip_width"] = self._dist(
                snap.left_hip,
                snap.right_hip
            )

        # ------------------------------------------------------------ #
        # Body Scale
        # ------------------------------------------------------------ #

        if (
            snap.mid_shoulder and
            snap.mid_hip
        ):

            dx = (
                snap.mid_shoulder[0] -
                snap.mid_hip[0]
            )

            dy = (
                snap.mid_shoulder[1] -
                snap.mid_hip[1]
            )

            snap.body_scale = float(
                np.hypot(dx, dy)
            )

        # ------------------------------------------------------------ #
        # Symmetry
        # ------------------------------------------------------------ #

        s = snap.symmetry

        lk = a.get("left_knee", self.MISSING)
        rk = a.get("right_knee", self.MISSING)

        if lk > 0 and rk > 0:

            s["knee_symmetry"] = abs(lk - rk)

        le = a.get("left_elbow", self.MISSING)
        re = a.get("right_elbow", self.MISSING)

        if le > 0 and re > 0:

            s["elbow_symmetry"] = abs(le - re)

        # ------------------------------------------------------------ #
        # Stability
        # ------------------------------------------------------------ #

        coords = []

        for lm in visible:

            coords.append(lm.x)
            coords.append(lm.y)

        if coords and snap.body_scale > 1e-6:

            raw_std = float(np.std(coords))

            snap.stability_score = raw_std / snap.body_scale

        # ------------------------------------------------------------ #
        # Balance Score
        # ------------------------------------------------------------ #

        if (
            snap.center_of_mass and
            snap.mid_hip
        ):

            dx = abs(
                snap.center_of_mass[0] -
                snap.mid_hip[0]
            )

            snap.balance_score = max(
                0.0,
                1.0 - (dx / max(snap.body_scale, 1))
            )

        # ------------------------------------------------------------ #
        # Velocities + Accelerations
        # ------------------------------------------------------------ #

        for key, val in a.items():

            if val < 0:
                continue

            prev_angle = self._prev_angles.get(key)

            if prev_angle is not None:

                velocity = val - prev_angle

            else:

                velocity = 0.0

            snap.velocities[key] = velocity

            prev_velocity = self._prev_velocities.get(key)

            if prev_velocity is not None:

                acceleration = (
                    velocity - prev_velocity
                )

            else:

                acceleration = 0.0

            snap.accelerations[key] = acceleration

            self._prev_angles[key] = val

            self._prev_velocities[key] = velocity

        # ------------------------------------------------------------ #
        # Posture Analysis
        # ------------------------------------------------------------ #

        p = snap.posture

        if (
            "neck_tilt" in a and
            a["neck_tilt"] > 20
        ):

            p["forward_head"] = 1.0

        else:

            p["forward_head"] = 0.0

        if (
            s.get("shoulder_symmetry", 0) > 15
        ):

            p["uneven_shoulders"] = 1.0

        else:

            p["uneven_shoulders"] = 0.0

        # ------------------------------------------------------------ #
        # Fatigue Estimation
        # ------------------------------------------------------------ #

        f = snap.fatigue

        movement_energy = np.mean(

            list(snap.velocities.values())

        ) if snap.velocities else 0

        instability = snap.stability_score

        fatigue_score = (

            abs(movement_energy) * 0.4 +
            instability * 0.6

        )

        f["fatigue_score"] = float(
            fatigue_score
        )

        # ------------------------------------------------------------ #
        # Temporal Smoothing
        # ------------------------------------------------------------ #

        self._history.append(snap)

        # ------------------------------------------------------------ #
        # Visibility
        # ------------------------------------------------------------ #

        required = [

            snap.left_shoulder,
            snap.right_shoulder,

            snap.left_hip,
            snap.right_hip,

            snap.left_knee,
            snap.right_knee,

        ]

        snap.fully_visible = all(
            j is not None
            for j in required
        )

        return snap

    # ================================================================= #
    # HELPERS
    # ================================================================= #

    @staticmethod
    def _midpoint(
        a: Landmark,
        b: Landmark
    ):

        return (

            int((a.px + b.px) / 2),
            int((a.py + b.py) / 2),

        )

    # ----------------------------------------------------------------- #

    @staticmethod
    def _dist(
        a: Landmark,
        b: Landmark
    ):

        return float(

            np.hypot(
                a.px - b.px,
                a.py - b.py
            )

        )

    # ----------------------------------------------------------------- #

    @staticmethod
    def _angle3(
        a,
        b,
        c
    ):

        if (
            a is None or
            b is None or
            c is None
        ):
            return -1.0

        return PoseEngine.landmark_angle(
            a,
            b,
            c
        )

    # ----------------------------------------------------------------- #

    @staticmethod
    def _angle3_3d(
        a,
        b,
        c
    ):

        if (
            a is None or
            b is None or
            c is None
        ):
            return -1.0

        return PoseEngine.landmark_angle_3d(
            a,
            b,
            c
        )

    # ----------------------------------------------------------------- #

    @staticmethod
    def _neck_tilt(
        nose,
        l_shoulder,
        r_shoulder,
    ):

        mid_x = (
            l_shoulder.px +
            r_shoulder.px
        ) / 2

        mid_y = (
            l_shoulder.py +
            r_shoulder.py
        ) / 2

        dx = nose.px - mid_x

        dy = mid_y - nose.py

        angle = np.degrees(

            np.arctan2(
                abs(dx),
                max(dy, 1e-9)
            )

        )

        return float(angle)

    # ================================================================= #
    # ML FEATURE VECTOR
    # ================================================================= #

    def extract_angle_vector(
        self,
        snap: BodySnapshot
    ) -> np.ndarray:

        keys = [

            "left_elbow",
            "right_elbow",

            "left_knee",
            "right_knee",

            "left_shoulder",
            "right_shoulder",

            "left_hip",
            "right_hip",

            "neck_tilt",
        ]

        def _safe(v):

            return (
                np.nan
                if v < 0
                else v
            )

        return np.array(

            [

                _safe(
                    snap.angles.get(
                        k,
                        self.MISSING
                    )
                )

                for k in keys

            ],

            dtype=np.float32
        )