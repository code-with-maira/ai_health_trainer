"""
movement_smoothing.py
---------------------
Advanced temporal smoothing filters for AI pose systems.

Features:
- Window smoothing
- EMA smoothing
- OneEuro filtering
- Kalman filtering
- Median filtering
- Landmark smoothing
- Angle smoothing
- Velocity estimation
- Acceleration estimation
"""

import math
import time
import numpy as np

from copy import copy
from collections import deque
from typing import Dict, List, Optional

from core.landmark_extractor import BodySnapshot

try:
    from core.pose_engine import Landmark
except ImportError:
    Landmark = object


# ================================================================== #
# MovementSmoothing
# ================================================================== #

class MovementSmoothing:
    """
    Rolling-window smoothing for Landmark objects.
    """

    def __init__(self, window_size: int = 5):

        self.window_size = window_size

        self._history = deque(maxlen=window_size)

    def smooth(self, landmarks: list) -> list:

        if not landmarks:
            return landmarks

        self._history.append(landmarks)

        # Not enough history yet
        if len(self._history) < 2:
            return landmarks

        smoothed = []

        for i, current_lm in enumerate(landmarks):

            frames_with_lm = [
                frame[i]
                for frame in self._history
                if i < len(frame)
            ]

            if not frames_with_lm:
                smoothed.append(current_lm)
                continue

            avg_x = float(np.mean([lm.x for lm in frames_with_lm]))
            avg_y = float(np.mean([lm.y for lm in frames_with_lm]))
            avg_z = float(np.mean([lm.z for lm in frames_with_lm]))

            avg_px = int(np.mean([lm.px for lm in frames_with_lm]))
            avg_py = int(np.mean([lm.py for lm in frames_with_lm]))

            lm_smooth = copy(current_lm)

            lm_smooth.x = avg_x
            lm_smooth.y = avg_y
            lm_smooth.z = avg_z

            lm_smooth.px = avg_px
            lm_smooth.py = avg_py

            smoothed.append(lm_smooth)

        return smoothed

    def reset(self):

        self._history.clear()


# ================================================================== #
# EMA
# ================================================================== #

class ExponentialMovingAverage:

    def __init__(self, alpha: float = 0.3):

        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1")

        self.alpha = alpha

        self._value: Optional[float] = None

    def update(self, x: float) -> float:

        if self._value is None:
            self._value = x

        else:
            self._value = (
                self.alpha * x +
                (1.0 - self.alpha) * self._value
            )

        return self._value

    def reset(self):

        self._value = None


# ================================================================== #
# Internal LowPass Filter
# ================================================================== #

class _LowPassFilter:

    def __init__(self, alpha: float):

        self._alpha = alpha

        self._value: Optional[float] = None

    @property
    def initialized(self):

        return self._value is not None

    def last_value(self) -> float:

        return self._value if self._value is not None else 0.0

    def filter(
        self,
        x: float,
        alpha: Optional[float] = None
    ) -> float:

        a = alpha if alpha is not None else self._alpha

        if self._value is None:
            self._value = x

        else:
            self._value = (
                a * x +
                (1.0 - a) * self._value
            )

        return self._value


# ================================================================== #
# OneEuroFilter
# ================================================================== #

class OneEuroFilter:

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):

        self.freq = freq

        self.min_cutoff = min_cutoff

        self.beta = beta

        self.d_cutoff = d_cutoff

        self._x_filt = _LowPassFilter(
            self._alpha(min_cutoff)
        )

        self._dx_filt = _LowPassFilter(
            self._alpha(d_cutoff)
        )

        self._last_time: Optional[float] = None

    def __call__(
        self,
        x: float,
        timestamp: Optional[float] = None
    ) -> float:

        t = timestamp if timestamp is not None else time.time()

        if self._last_time is not None:

            dt = max(t - self._last_time, 1e-6)

            self.freq = min(240.0, 1.0 / dt)

        self._last_time = t

        prev = (
            self._x_filt.last_value()
            if self._x_filt.initialized
            else x
        )

        dx = (x - prev) * self.freq

        edx = self._dx_filt.filter(
            dx,
            self._alpha(self.d_cutoff)
        )

        cutoff = (
            self.min_cutoff +
            self.beta * abs(edx)
        )

        return self._x_filt.filter(
            x,
            self._alpha(cutoff)
        )

    def _alpha(self, cutoff: float) -> float:

        tau = 1.0 / (2.0 * math.pi * cutoff)

        te = 1.0 / self.freq

        return 1.0 / (1.0 + tau / te)


# ================================================================== #
# KalmanFilter1D
# ================================================================== #

class KalmanFilter1D:

    def __init__(
        self,
        Q: float = 1e-3,
        R: float = 0.1
    ):

        self.Q = Q

        self.R = R

        self._x: Optional[float] = None

        self._P = 1.0

    def update(self, z: float) -> float:

        if self._x is None:

            self._x = z

            return z

        P_pred = self._P + self.Q

        K = P_pred / (P_pred + self.R)

        self._x = self._x + K * (z - self._x)

        self._P = (1.0 - K) * P_pred

        return self._x

    def reset(self):

        self._x = None

        self._P = 1.0


# ================================================================== #
# MedianFilter
# ================================================================== #

class MedianFilter:

    def __init__(self, window: int = 5):

        self._buf = deque(maxlen=window)

    def update(self, x: float) -> float:

        self._buf.append(x)

        return float(np.median(self._buf))


# ================================================================== #
# LandmarkSmoother
# ================================================================== #

class LandmarkSmoother:

    def __init__(
        self,
        num_landmarks: int = 33,
        **kwargs
    ):

        self._x = [
            OneEuroFilter(**kwargs)
            for _ in range(num_landmarks)
        ]

        self._y = [
            OneEuroFilter(**kwargs)
            for _ in range(num_landmarks)
        ]

    def smooth(self, landmarks: list) -> list:

        for lm in landmarks:

            if not hasattr(lm, "index"):
                continue

            if lm.index >= len(self._x):
                continue

            lm.x = self._x[lm.index](lm.x)

            lm.y = self._y[lm.index](lm.y)

        return landmarks


# ================================================================== #
# AngleSmoother
# ================================================================== #

class AngleSmoother:

    ANGLE_KEYS = [

        "left_elbow",
        "right_elbow",

        "left_knee",
        "right_knee",

        "left_shoulder",
        "right_shoulder",

        "left_hip",
        "right_hip",

        "left_ankle",
        "right_ankle",

        "neck_tilt",

        "trunk_lean_left",
        "trunk_lean_right",
        "trunk_lean_avg",
    ]

    def __init__(
        self,
        filter_type: str = "one_euro",
        **kwargs
    ):

        self._filter_type = filter_type

        self._kwargs = kwargs

        self._filters = {
            key: self._make_filter()
            for key in self.ANGLE_KEYS
        }

    def _make_filter(self):

        ft = self._filter_type

        if ft == "one_euro":

            return OneEuroFilter(**self._kwargs)

        elif ft == "ema":

            return ExponentialMovingAverage(
                self._kwargs.get("alpha", 0.3)
            )

        elif ft == "kalman":

            return KalmanFilter1D(
                Q=self._kwargs.get("Q", 1e-3),
                R=self._kwargs.get("R", 0.1),
            )

        elif ft == "median":

            return MedianFilter(
                self._kwargs.get("window", 5)
            )

        else:
            raise ValueError(
                f"Unknown filter type: {ft}"
            )

    def smooth(
        self,
        snap: BodySnapshot
    ) -> BodySnapshot:

        for key, filt in self._filters.items():

            raw = snap.angles.get(key, -1.0)

            if raw < 0:
                continue

            if self._filter_type == "one_euro":

                snap.angles[key] = filt(raw)

            else:

                snap.angles[key] = filt.update(raw)

        return snap


# ================================================================== #
# VelocityEstimator
# ================================================================== #

class VelocityEstimator:

    def __init__(self, window: int = 5):

        self._window = window

        self._history: Dict[str, deque] = {}

    def update(
        self,
        angles: Dict[str, float]
    ) -> Dict[str, float]:

        velocities = {}

        for key, val in angles.items():

            if val < 0:
                continue

            if key not in self._history:

                self._history[key] = deque(
                    maxlen=self._window
                )

            buf = self._history[key]

            buf.append(val)

            if len(buf) >= 2:

                diffs = np.diff(buf)

                velocities[key] = float(
                    np.mean(diffs)
                )

            else:

                velocities[key] = 0.0

        return velocities

    def acceleration(
        self,
        angles: Dict[str, float]
    ) -> Dict[str, float]:

        accel = {}

        for key, val in angles.items():

            if val < 0:
                continue

            buf = self._history.get(key)

            if buf and len(buf) >= 3:

                accel[key] = float(
                    (buf[-1] - buf[-2]) -
                    (buf[-2] - buf[-3])
                )

            else:

                accel[key] = 0.0

        return accel

    def reset(self):

        self._history.clear()