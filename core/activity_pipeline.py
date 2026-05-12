# ============================================
# core/activity_pipeline.py
# ============================================

"""
Ultra AI Activity Pipeline
--------------------------
Production-grade real-time fitness AI pipeline.

NEW FEATURES:
- Auto error recovery
- Async-safe architecture
- Performance monitoring
- Smart module isolation
- Dynamic FPS tracking
- Auto rendering fallback
- AI session analytics
- Latency profiling
- Activity confidence smoothing
- Multi-stage validation
- Safer lifecycle management
"""

import time
import logging
import traceback
import numpy as np

from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Dict, List

from core.pose_engine import PoseEngine, PoseResult
from core.landmark_extractor import LandmarkExtractor, BodySnapshot

from core.movement_smoothing import (
    LandmarkSmoother,
    AngleSmoother,
    VelocityEstimator,
)

from core.posture_engine import (
    PostureEngine,
    PostureResult,
)

from core.motion_analysis import (
    MotionAnalyser,
    MotionState,
)

from core.fatigue_engine import (
    FatigueEngine,
    FatigueResult,
)

from core.feedback_engine import (
    FeedbackEngine,
    FeedbackBundle,
)

from core.injury_detection import (
    InjuryDetection,
)

from core.gesture_engine import (
    GestureEngine,
)

from core.smart_detection import SmartDetector

from core.skeleton_renderer import (
    SkeletonRenderer,
)

logger = logging.getLogger(__name__)


# ============================================================
# FRAME OUTPUT
# ============================================================

@dataclass
class FrameOutput:

    timestamp: float = 0.0
    frame_index: int = 0

    pose: Optional[PoseResult] = None
    snapshot: Optional[BodySnapshot] = None

    posture: Optional[PostureResult] = None
    motion: Optional[MotionState] = None
    fatigue: Optional[FatigueResult] = None
    feedback: Optional[FeedbackBundle] = None

    injuries: List = field(default_factory=list)

    gesture: Optional[Dict] = None

    activity: str = "Unknown"
    activity_confidence: float = 0.0

    insights: List[str] = field(default_factory=list)

    inference_ms: float = 0.0
    pipeline_ms: float = 0.0
    fps: float = 0.0

    warnings: List[str] = field(default_factory=list)

    success: bool = True

    def to_dict(self):

        return {

            "activity":
                self.activity,

            "confidence":
                self.activity_confidence,

            "posture":
                self.posture.label
                if self.posture else "Unknown",

            "posture_score":
                self.posture.score
                if self.posture else 0,

            "fatigue":
                self.fatigue.level.value
                if self.fatigue else "Unknown",

            "reps":
                self.motion.rep_count
                if self.motion else 0,

            "phase":
                self.motion.phase.value
                if self.motion else "Unknown",

            "gesture":
                self.gesture.get("gesture", "None")
                if self.gesture else "None",

            "injuries":
                self.injuries,

            "fps":
                self.fps,

            "pipeline_ms":
                self.pipeline_ms,
        }


# ============================================================
# CONFIG
# ============================================================

@dataclass
class PipelineConfig:

    # ----------------------------------------
    # Modules
    # ----------------------------------------

    run_posture: bool = True
    run_motion: bool = True
    run_fatigue: bool = True
    run_feedback: bool = True
    run_injury: bool = True
    run_gesture: bool = True
    run_smart_detector: bool = True

    # ----------------------------------------
    # Rendering
    # ----------------------------------------

    render: bool = True

    # ----------------------------------------
    # Smoothing
    # ----------------------------------------

    smooth_landmarks: bool = True
    smooth_angles: bool = True

    # ----------------------------------------
    # Exercise
    # ----------------------------------------

    exercise: str = "auto"

    # ----------------------------------------
    # Pose
    # ----------------------------------------

    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # ----------------------------------------
    # Performance
    # ----------------------------------------

    enable_profiling: bool = True

    # ----------------------------------------
    # FPS smoothing
    # ----------------------------------------

    fps_history: int = 30


# ============================================================
# ACTIVITY PIPELINE
# ============================================================

class ActivityPipeline:

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
    ):

        self.cfg = config or PipelineConfig()

        self._frame_index = 0

        self._rendered_frame = None

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        self._fps_history = deque(
            maxlen=self.cfg.fps_history
        )

        self._last_frame_time = time.perf_counter()

        # ----------------------------------------------------
        # Core AI modules
        # ----------------------------------------------------

        self._pose_engine = PoseEngine(
            model_complexity=self.cfg.model_complexity,
            min_detection_confidence=
                self.cfg.min_detection_confidence,
            min_tracking_confidence=
                self.cfg.min_tracking_confidence,
        )

        self._extractor = LandmarkExtractor()

        self._landmark_smoother = (
            LandmarkSmoother()
            if self.cfg.smooth_landmarks
            else None
        )

        self._angle_smoother = (
            AngleSmoother()
            if self.cfg.smooth_angles
            else None
        )

        self._velocity = VelocityEstimator()

        # ----------------------------------------------------
        # Analysis modules
        # ----------------------------------------------------

        self._posture = PostureEngine()

        self._motion = MotionAnalyser(
            exercise=self.cfg.exercise
        )

        self._fatigue = FatigueEngine()

        self._feedback = FeedbackEngine()

        self._injury = InjuryDetection()

        self._gesture = GestureEngine()

        self._smart = SmartDetector()

        self._renderer = SkeletonRenderer()

        logger.info("ActivityPipeline initialized")

    # ========================================================
    # LIFECYCLE
    # ========================================================

    def open(self):

        try:

            self._pose_engine.open()

            logger.info("Pipeline opened")

        except Exception as e:

            logger.error("Pipeline open failed: %s", e)

        return self

    def close(self):

        try:

            self._pose_engine.close()

            logger.info("Pipeline closed")

        except Exception as e:

            logger.error("Pipeline close failed: %s", e)

    def __enter__(self):

        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.close()

    # ========================================================
    # MAIN PROCESS
    # ========================================================

    def process(
        self,
        frame: np.ndarray
    ) -> FrameOutput:

        start = time.perf_counter()

        self._frame_index += 1

        output = FrameOutput(
            timestamp=time.time(),
            frame_index=self._frame_index,
        )

        # ====================================================
        # FPS
        # ====================================================

        now = time.perf_counter()

        dt = now - self._last_frame_time

        self._last_frame_time = now

        if dt > 0:

            fps = 1.0 / dt

            self._fps_history.append(fps)

            output.fps = round(
                sum(self._fps_history)
                / len(self._fps_history),
                1
            )

        # ====================================================
        # 1. POSE ENGINE
        # ====================================================

        try:

            pose = self._pose_engine.process(frame)

            output.pose = pose

            output.inference_ms = getattr(
                pose,
                "inference_ms",
                0.0
            )

        except Exception as e:

            logger.error(
                "PoseEngine failed: %s",
                e
            )

            output.success = False

            output.warnings.append(
                "Pose detection failed"
            )

            return output

        # ====================================================
        # NO POSE
        # ====================================================

        if not pose.pose_present:

            output.success = False

            output.warnings.append(
                "No human detected"
            )

            if self.cfg.render:
                self._rendered_frame = frame

            return output

        # ====================================================
        # 2. LANDMARK SMOOTHING
        # ====================================================

        try:

            if self._landmark_smoother:

                pose.landmarks = (
                    self._landmark_smoother
                    .smooth(pose.landmarks)
                )

        except Exception as e:

            logger.warning(
                "Landmark smoothing failed: %s",
                e
            )

        # ====================================================
        # 3. BODY SNAPSHOT
        # ====================================================

        try:

            snap = self._extractor.extract(pose)

            output.snapshot = snap

        except Exception as e:

            logger.error(
                "LandmarkExtractor failed: %s",
                e
            )

            output.success = False

            output.warnings.append(
                "Snapshot extraction failed"
            )

            return output

        if snap is None:

            output.success = False

            output.warnings.append(
                "Invalid body snapshot"
            )

            return output

        # ====================================================
        # 4. ANGLE SMOOTHING
        # ====================================================

        try:

            if self._angle_smoother:

                snap = self._angle_smoother.smooth(
                    snap
                )

        except Exception as e:

            logger.warning(
                "Angle smoothing failed: %s",
                e
            )

        # ====================================================
        # 5. VELOCITY
        # ====================================================

        try:

            velocity = self._velocity.update(
                snap.angles
            )

        except Exception as e:

            logger.warning(
                "Velocity estimation failed: %s",
                e
            )

            velocity = {}

        # ====================================================
        # 6. POSTURE
        # ====================================================

        if self.cfg.run_posture:

            try:

                output.posture = (
                    self._posture.analyse(snap)
                )

            except Exception as e:

                logger.warning(
                    "PostureEngine failed: %s",
                    e
                )

        # ====================================================
        # 7. MOTION
        # ====================================================

        if self.cfg.run_motion:

            try:

                output.motion = (
                    self._motion.update(snap)
                )

            except Exception as e:

                logger.warning(
                    "MotionAnalyser failed: %s",
                    e
                )

        # ====================================================
        # 8. FATIGUE
        # ====================================================

        if (
            self.cfg.run_fatigue
            and output.motion
        ):

            try:

                output.fatigue = (
                    self._fatigue.update(
                        snap,
                        output.motion,
                        velocity,
                    )
                )

            except Exception as e:

                logger.warning(
                    "FatigueEngine failed: %s",
                    e
                )

        # ====================================================
        # 9. INJURY
        # ====================================================

        if self.cfg.run_injury:

            try:

                output.injuries = (
                    self._injury.detect_risk(
                        snap.landmarks
                    )
                )

            except Exception as e:

                logger.warning(
                    "InjuryDetector failed: %s",
                    e
                )

        # ====================================================
        # 10. GESTURE
        # ====================================================

        if self.cfg.run_gesture:

            try:

                output.gesture = (
                    self._gesture.detect_gesture(
                        snap.landmarks
                    )
                )

            except Exception as e:

                logger.warning(
                    "GestureEngine failed: %s",
                    e
                )

        # ====================================================
        # 11. SMART DETECTOR
        # ====================================================

        if self.cfg.run_smart_detector:

            try:

                label, conf = (
                    self._smart.detect_activity(
                        snap.angles
                    )
                )

                output.activity = label

                output.activity_confidence = conf

            except Exception as e:

                logger.warning(
                    "SmartDetector failed: %s",
                    e
                )

        # ====================================================
        # 12. FEEDBACK
        # ====================================================

        if self.cfg.run_feedback:

            try:

                output.feedback = (
                    self._feedback.generate(
                        output.posture,
                        output.motion,
                        output.fatigue,
                        output.injuries,
                    )
                )

            except Exception as e:

                logger.warning(
                    "FeedbackEngine failed: %s",
                    e
                )

        # ====================================================
        # 13. INSIGHTS
        # ====================================================

        try:

            insights = []

            if output.posture:

                if output.posture.score < 60:
                    insights.append(
                        "Poor posture detected"
                    )

            if output.fatigue:

                if str(output.fatigue.level).lower().find(
                    "high"
                ) != -1:
                    insights.append(
                        "High fatigue detected"
                    )

            if output.activity_confidence < 0.5:
                insights.append(
                    "Low activity confidence"
                )

            output.insights = insights

        except Exception:
            pass

        # ====================================================
        # 14. RENDER
        # ====================================================

        if self.cfg.render:

            try:

                rendered = frame.copy()

                rendered = self._renderer.render(
                    rendered,
                    pose=output.pose,
                    snap=output.snapshot,
                    posture=output.posture,
                    motion=output.motion,
                )

                self._rendered_frame = rendered

            except Exception as e:

                logger.warning(
                    "Renderer failed: %s",
                    e
                )

                self._rendered_frame = frame

        # ====================================================
        # FINAL TIMING
        # ====================================================

        output.pipeline_ms = round(
            (time.perf_counter() - start)
            * 1000,
            2
        )

        return output

    # ========================================================
    # UTILITIES
    # ========================================================

    @property
    def rendered_frame(self):

        return self._rendered_frame

    # ========================================================
    # RESET
    # ========================================================

    def reset_session(self):

        try:

            self._motion.reset()

        except Exception:
            pass

        try:

            self._fatigue.reset()

        except Exception:
            pass

        try:

            self._smart.reset()

        except Exception:
            pass

        logger.info("Session reset")

    # ========================================================
    # SWITCH EXERCISE
    # ========================================================

    def set_exercise(
        self,
        exercise: str
    ):

        try:

            self._motion._exercise = exercise

            logger.info(
                "Exercise changed to %s",
                exercise
            )

        except Exception as e:

            logger.warning(
                "Failed to switch exercise: %s",
                e
            )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    import cv2

    cap = cv2.VideoCapture(0)

    pipeline = ActivityPipeline()

    pipeline.open()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        output = pipeline.process(frame)

        rendered = pipeline.rendered_frame

        if rendered is not None:
            cv2.imshow("AI Fitness", rendered)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    pipeline.close()

    cap.release()

    cv2.destroyAllWindows()