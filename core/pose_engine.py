"""
pose_engine.py
--------------
Advanced MediaPipe Pose wrapper for AI fitness systems.

IMPROVEMENTS:
- Fixed '_pose' AttributeError
- Better resource management
- FPS smoothing
- Safe reopen/close
- Optional segmentation
- Improved visibility handling
- Stable confidence scoring
- Better error handling
- Thread-safe style state management
"""

import cv2
import time
import logging
import mediapipe as mp
import numpy as np

from dataclasses import dataclass, field
from typing import Optional, List, Dict


logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ===================================================================== #
# LANDMARK NAMES
# ===================================================================== #

LANDMARK_NAMES = {

    0: "NOSE",

    1: "LEFT_EYE_INNER",
    2: "LEFT_EYE",
    3: "LEFT_EYE_OUTER",

    4: "RIGHT_EYE_INNER",
    5: "RIGHT_EYE",
    6: "RIGHT_EYE_OUTER",

    7: "LEFT_EAR",
    8: "RIGHT_EAR",

    9: "MOUTH_LEFT",
    10: "MOUTH_RIGHT",

    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",

    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",

    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",

    17: "LEFT_PINKY",
    18: "RIGHT_PINKY",

    19: "LEFT_INDEX",
    20: "RIGHT_INDEX",

    21: "LEFT_THUMB",
    22: "RIGHT_THUMB",

    23: "LEFT_HIP",
    24: "RIGHT_HIP",

    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",

    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",

    29: "LEFT_HEEL",
    30: "RIGHT_HEEL",

    31: "LEFT_FOOT_INDEX",
    32: "RIGHT_FOOT_INDEX",
}


# ===================================================================== #
# LANDMARK
# ===================================================================== #

@dataclass
class Landmark:

    index: int
    name: str

    x: float
    y: float
    z: float

    visibility: float

    px: int = 0
    py: int = 0

    @property
    def point2d(self):
        return (self.px, self.py)

    @property
    def point3d(self):
        return (self.x, self.y, self.z)

    @property
    def is_visible(self):
        return self.visibility >= 0.5


# ===================================================================== #
# RESULT
# ===================================================================== #

@dataclass
class PoseResult:

    pose_present: bool = False

    landmarks: List[Landmark] = field(default_factory=list)

    landmark_map: Dict[str, Landmark] = field(default_factory=dict)

    index_map: Dict[int, Landmark] = field(default_factory=dict)

    inference_ms: float = 0.0

    fps: float = 0.0

    confidence: float = 0.0

    raw_result: Optional[object] = None

    segmentation_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------ #

    def get(
        self,
        index: int
    ) -> Optional[Landmark]:

        return self.index_map.get(index)

    # ------------------------------------------------------------ #

    def get_by_name(
        self,
        name: str
    ) -> Optional[Landmark]:

        return self.landmark_map.get(
            name.upper()
        )


# ===================================================================== #
# POSE ENGINE
# ===================================================================== #

class PoseEngine:

    MIN_VISIBILITY = 0.5

    # ------------------------------------------------------------ #

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smooth_landmarks: bool = True,
        static_image_mode: bool = False,
        enable_segmentation: bool = False,
    ):

        self.cfg = dict(

            static_image_mode=static_image_mode,

            model_complexity=model_complexity,

            smooth_landmarks=smooth_landmarks,

            enable_segmentation=enable_segmentation,

            min_detection_confidence=min_detection_confidence,

            min_tracking_confidence=min_tracking_confidence,
        )

        # ========================================================= #
        # FIXED: _pose created properly
        # ========================================================= #

        self._pose = None

        self._create_pose()

        # FPS smoothing
        self._last_frame_time = None

        self._fps_history = []

        logger.info(
            "PoseEngine initialized"
        )

    # ================================================================= #
    # INTERNAL
    # ================================================================= #

    def _create_pose(self):

        if self._pose is not None:
            return

        self._pose = mp_pose.Pose(
            **self.cfg
        )

    # ================================================================= #
    # RESOURCE MANAGEMENT
    # ================================================================= #

    def open(self):

        if self._pose is None:

            self._create_pose()

            logger.info(
                "PoseEngine reopened"
            )

        return self

    # ------------------------------------------------------------ #

    def close(self):

        if self._pose is not None:

            self._pose.close()

            self._pose = None

            logger.info(
                "PoseEngine closed"
            )

    # ------------------------------------------------------------ #

    def reset(self):

        self.close()

        self.open()

        self._last_frame_time = None

        self._fps_history.clear()

    # ------------------------------------------------------------ #

    def __enter__(self):

        self.open()

        return self

    # ------------------------------------------------------------ #

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.close()

    # ================================================================= #
    # PROCESS
    # ================================================================= #

    def process(
        self,
        frame: np.ndarray
    ) -> PoseResult:

        if frame is None:

            raise ValueError(
                "Frame is None"
            )

        if self._pose is None:

            raise RuntimeError(
                "PoseEngine is closed"
            )

        h, w = frame.shape[:2]

        # ------------------------------------------------------------ #
        # RGB conversion
        # ------------------------------------------------------------ #

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb.flags.writeable = False

        # ------------------------------------------------------------ #
        # Inference
        # ------------------------------------------------------------ #

        t0 = time.perf_counter()

        raw = self._pose.process(rgb)

        inference_ms = (
            time.perf_counter() - t0
        ) * 1000

        # ------------------------------------------------------------ #
        # FPS
        # ------------------------------------------------------------ #

        now = time.perf_counter()

        if self._last_frame_time is None:

            fps = 0.0

        else:

            dt = max(
                now - self._last_frame_time,
                1e-6
            )

            fps = 1.0 / dt

        self._last_frame_time = now

        # Smooth FPS
        self._fps_history.append(fps)

        if len(self._fps_history) > 10:
            self._fps_history.pop(0)

        fps = float(
            np.mean(self._fps_history)
        )

        # ------------------------------------------------------------ #
        # NO POSE
        # ------------------------------------------------------------ #

        if not raw.pose_landmarks:

            return PoseResult(

                pose_present=False,

                inference_ms=round(
                    inference_ms,
                    2
                ),

                fps=round(fps, 1),

                raw_result=raw,
            )

        # ------------------------------------------------------------ #
        # LANDMARK EXTRACTION
        # ------------------------------------------------------------ #

        landmarks = []

        visibilities = []

        for idx, lm in enumerate(
            raw.pose_landmarks.landmark
        ):

            visibilities.append(
                lm.visibility
            )

            if lm.visibility < self.MIN_VISIBILITY:
                continue

            landmarks.append(

                Landmark(

                    index=idx,

                    name=LANDMARK_NAMES.get(
                        idx,
                        f"LM_{idx}"
                    ),

                    x=float(lm.x),

                    y=float(lm.y),

                    z=float(lm.z),

                    visibility=float(
                        lm.visibility
                    ),

                    px=int(lm.x * w),

                    py=int(lm.y * h),
                )
            )

        # ------------------------------------------------------------ #
        # MAPS
        # ------------------------------------------------------------ #

        landmark_map = {
            lm.name: lm
            for lm in landmarks
        }

        index_map = {
            lm.index: lm
            for lm in landmarks
        }

        # ------------------------------------------------------------ #
        # CONFIDENCE
        # ------------------------------------------------------------ #

        confidence = float(
            np.mean(visibilities)
        )

        # ------------------------------------------------------------ #
        # Segmentation
        # ------------------------------------------------------------ #

        seg_mask = None

        if (
            hasattr(raw, "segmentation_mask")
            and raw.segmentation_mask is not None
        ):

            seg_mask = raw.segmentation_mask

        # ------------------------------------------------------------ #
        # RESULT
        # ------------------------------------------------------------ #

        return PoseResult(

            pose_present=True,

            landmarks=landmarks,

            landmark_map=landmark_map,

            index_map=index_map,

            inference_ms=round(
                inference_ms,
                2
            ),

            fps=round(fps, 1),

            confidence=round(
                confidence,
                2
            ),

            raw_result=raw,

            segmentation_mask=seg_mask,
        )

    # ================================================================= #
    # DRAWING
    # ================================================================= #

    def draw_pose(
        self,
        frame: np.ndarray,
        result: PoseResult,
        draw_connections: bool = True,
    ) -> np.ndarray:

        if (
            not result.pose_present or
            not result.raw_result
        ):
            return frame

        connections = (
            mp_pose.POSE_CONNECTIONS
            if draw_connections
            else None
        )

        mp_drawing.draw_landmarks(

            frame,

            result.raw_result.pose_landmarks,

            connections,
        )

        return frame

    # ================================================================= #
    # ANGLE HELPERS
    # ================================================================= #

    @staticmethod
    def landmark_angle(
        a: Landmark,
        b: Landmark,
        c: Landmark,
    ) -> float:

        ba = np.array([
            a.px - b.px,
            a.py - b.py
        ], dtype=np.float32)

        bc = np.array([
            c.px - b.px,
            c.py - b.py
        ], dtype=np.float32)

        denom = (
            np.linalg.norm(ba) *
            np.linalg.norm(bc)
        )

        if denom < 1e-6:
            return 0.0

        cos_angle = np.clip(
            np.dot(ba, bc) / denom,
            -1.0,
            1.0
        )

        return float(
            np.degrees(
                np.arccos(cos_angle)
            )
        )

    # ------------------------------------------------------------ #

    @staticmethod
    def landmark_angle_3d(
        a: Landmark,
        b: Landmark,
        c: Landmark,
    ) -> float:

        ba = np.array([
            a.x - b.x,
            a.y - b.y,
            a.z - b.z
        ], dtype=np.float32)

        bc = np.array([
            c.x - b.x,
            c.y - b.y,
            c.z - b.z
        ], dtype=np.float32)

        denom = (
            np.linalg.norm(ba) *
            np.linalg.norm(bc)
        )

        if denom < 1e-6:
            return 0.0

        cos_angle = np.clip(
            np.dot(ba, bc) / denom,
            -1.0,
            1.0
        )

        return float(
            np.degrees(
                np.arccos(cos_angle)
            )
        )

    # ================================================================= #
    # UTILITIES
    # ================================================================= #

    @staticmethod
    def normalize_point(
        lm: Landmark,
        width: int,
        height: int
    ):

        return (
            int(lm.x * width),
            int(lm.y * height)
        )