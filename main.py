# =========================================================
# main.py (IMPROVED + STABLE + SAFE VERSION)
# =========================================================
# =========================================================
# main.py
# =========================================================

import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="google.protobuf.symbol_database"
)

import sys
import time
import logging
import traceback

import cv2

from core.camera_manager import CameraManager
from core.pose_engine import PoseEngine
from core.landmark_extractor import LandmarkExtractor

from core.motion_analysis import MotionAnalyser
from core.feedback_engine import FeedbackEngine
from core.skeleton_renderer import SkeletonRenderer
from core.smart_detection import SmartDetector

from core.posture_engine import PostureEngine
from core.fatigue_engine import FatigueEngine
from core.gesture_engine import GestureEngine
from core.injury_detection import InjuryDetection

from core.movement_smoothing import (
    LandmarkSmoother,
    AngleSmoother,
)

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

# =========================================================
# SETTINGS
# =========================================================

WINDOW_NAME = "AI Health Trainer"

TARGET_FPS = 30

FRAME_TIME = 1 / TARGET_FPS

# =========================================================
# UI HELPERS
# =========================================================

def draw_text(
    frame,
    text,
    pos,
    scale=0.65,
    color=(255, 255, 255),
    thickness=2,
):

    cv2.putText(
        frame,
        str(text),
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(
    frame,
    y1,
    y2,
    alpha=0.55,
):

    h, w = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, y1),
        (w, y2),
        (20, 20, 20),
        -1,
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame,
    )

# =========================================================
# SAFE HELPERS
# =========================================================

def safe_angle(value):

    try:

        value = float(value)

        if value < 0 or value > 180:
            return None

        return value

    except:
        return None


def safe_landmark_exists(
    landmarks,
    idx,
):

    try:

        if landmarks is None:
            return False

        if idx >= len(landmarks):
            return False

        lm = landmarks[idx]

        return (
            hasattr(lm, "x")
            and hasattr(lm, "y")
        )

    except:
        return False

# =========================================================
# MAIN
# =========================================================

def main():

    # =====================================================
    # INITIALIZATION
    # =====================================================

    try:

        camera = CameraManager(
            camera_index=0,
            width=1280,
            height=720,
            fps=30,
        )

        camera.open()

        pose_engine = PoseEngine()
        pose_engine.open()

        extractor = LandmarkExtractor()

        motion_analyser = MotionAnalyser()

        posture_engine = PostureEngine()

        fatigue_engine = FatigueEngine()

        gesture_engine = GestureEngine()

        injury_detector = InjuryDetection()

        feedback_engine = FeedbackEngine()

        renderer = SkeletonRenderer()

        smart_detector = SmartDetector()

        landmark_smoother = LandmarkSmoother()

        angle_smoother = AngleSmoother()

        logger.info(
            "AI Health Trainer Started"
        )

    except Exception as e:

        logger.exception(
            "Initialization failed: %s",
            e,
        )

        sys.exit(1)

    # =====================================================
    # FPS
    # =====================================================

    prev_time = time.time()

    fps = 0.0

    # =====================================================
    # MAIN LOOP
    # =====================================================

    while True:

        frame_start = time.time()

        # =================================================
        # DEFAULT VALUES
        # =================================================

        prediction = "Unknown"

        confidence = 0.0

        posture_label = "Unknown"

        posture_score = 0

        fatigue_text = "Normal"

        gesture_text = "None"

        injury_text = "Safe"

        reps = 0

        feedback = (
            "Stand clearly in front of camera"
        )

        fatigue = None

        motion = None

        posture = None

        snapshot = None

        # =================================================
        # CAMERA READ
        # =================================================

        try:

            frame = camera.read()

            if frame is None:

                logger.warning(
                    "Empty camera frame"
                )

                continue

            if not hasattr(frame, "shape"):

                logger.warning(
                    "Invalid frame"
                )

                continue

            # Mirror View
            frame = cv2.flip(
                frame,
                1,
            )

        except Exception as e:

            logger.exception(
                "Camera Error: %s",
                e,
            )

            break

        # =================================================
        # POSE DETECTION
        # =================================================

        try:

            pose = pose_engine.process(frame)

        except Exception as e:

            logger.exception(
                "PoseEngine Error: %s",
                e,
            )

            pose = None

        # =================================================
        # AI PIPELINE
        # =================================================

        if (
            pose is not None
            and hasattr(pose, "pose_present")
            and pose.pose_present
            and hasattr(pose, "landmarks")
            and pose.landmarks
        ):

            try:

                # =========================================
                # LANDMARK SMOOTHING
                # =========================================

                try:

                    pose.landmarks = (
                        landmark_smoother.smooth(
                            pose.landmarks
                        )
                    )

                except Exception as e:

                    logger.warning(
                        "Landmark smoothing failed: %s",
                        e,
                    )

                # =========================================
                # SNAPSHOT EXTRACTION
                # =========================================

                snapshot = extractor.extract(
                    pose
                )

                if snapshot is None:
                    continue

                if not hasattr(
                    snapshot,
                    "angles"
                ):

                    snapshot.angles = {}

                # =========================================
                # ANGLE SMOOTHING
                # =========================================

                try:

                    snapshot = (
                        angle_smoother.smooth(
                            snapshot
                        )
                    )

                except Exception as e:

                    logger.warning(
                        "Angle smoothing failed: %s",
                        e,
                    )

                # =========================================
                # MOTION ANALYSIS
                # =========================================

                try:

                    motion = (
                        motion_analyser.update(
                            snapshot
                        )
                    )

                    reps = getattr(
                        motion,
                        "rep_count",
                        0,
                    )

                except Exception as e:

                    logger.warning(
                        "MotionAnalyser failed: %s",
                        e,
                    )

                # =========================================
                # POSTURE ANALYSIS
                # =========================================

                try:

                    posture = (
                        posture_engine.analyse(
                            snapshot
                        )
                    )

                    posture_label = getattr(
                        posture.classification,
                        "value",
                        "Unknown",
                    )

                    posture_score = getattr(
                        posture,
                        "score",
                        0,
                    )

                except Exception as e:

                    logger.warning(
                        "PostureEngine failed: %s",
                        e,
                    )

                # =========================================
                # FATIGUE ANALYSIS
                # =========================================

                try:

                    elbow_angle = None

                    if (
                        hasattr(
                            snapshot,
                            "angles"
                        )
                        and isinstance(
                            snapshot.angles,
                            dict
                        )
                    ):

                        elbow_angle = safe_angle(
                            snapshot.angles.get(
                                "left_elbow"
                            )
                        )

                    if elbow_angle is not None:

                        fatigue = (
                            fatigue_engine.detect_fatigue(
                                {
                                    "elbow_angle":
                                    elbow_angle
                                }
                            )
                        )

                        if fatigue:

                            fatigue_text = str(
                                getattr(
                                    fatigue.level,
                                    "value",
                                    fatigue.level
                                )
                            )

                    else:

                        fatigue_text = "Unknown"

                except Exception as e:

                    logger.warning(
                        "FatigueEngine failed: %s",
                        e,
                    )

                # =========================================
                # GESTURE DETECTION
                # =========================================

                try:

                    gesture = None

                    valid = all([

                        safe_landmark_exists(
                            pose.landmarks,
                            11,
                        ),

                        safe_landmark_exists(
                            pose.landmarks,
                            13,
                        ),

                        safe_landmark_exists(
                            pose.landmarks,
                            15,
                        ),

                    ])

                    if valid:

                        if hasattr(
                            gesture_engine,
                            "detect"
                        ):

                            gesture = (
                                gesture_engine.detect(
                                    snapshot
                                )
                            )

                        elif hasattr(
                            gesture_engine,
                            "detect_gesture"
                        ):

                            gesture = (
                                gesture_engine.detect_gesture(
                                    pose.landmarks
                                )
                            )

                    if gesture:

                        if hasattr(
                            gesture,
                            "name"
                        ):

                            gesture_text = (
                                gesture.name
                            )

                        elif isinstance(
                            gesture,
                            dict
                        ):

                            gesture_text = (
                                gesture.get(
                                    "gesture",
                                    "Detected"
                                )
                            )

                        else:

                            gesture_text = str(
                                gesture
                            )

                except Exception as e:

                    logger.warning(
                        "GestureEngine failed: %s",
                        e,
                    )

                # =========================================
                # INJURY DETECTION
                # =========================================

                try:

                    required = [
                        11, 12,
                        23, 24,
                        25, 26,
                        27, 28
                    ]

                    valid = all([

                        safe_landmark_exists(
                            pose.landmarks,
                            idx
                        )

                        for idx in required
                    ])

                    if valid:

                        raw_landmarks = []

                        for lm in pose.landmarks:

                            raw_landmarks.append({

                                "x": float(lm.x),

                                "y": float(lm.y),

                            })

                        injury_result = (
                            injury_detector.detect_risk(
                                raw_landmarks
                            )
                        )

                        if injury_result:

                            injury_text = str(
                                getattr(
                                    injury_result,
                                    "risk_level",
                                    "Risk"
                                )
                            )

                except Exception as e:

                    logger.warning(
                        "InjuryDetection failed: %s",
                        e,
                    )

                # =========================================
                # ACTIVITY DETECTION
                # =========================================

                try:

                    prediction, confidence = (
                        smart_detector.detect_activity(
                            snapshot.angles
                        )
                    )

                except Exception as e:

                    logger.warning(
                        "SmartDetector failed: %s",
                        e,
                    )

                # =========================================
                # FEEDBACK ENGINE
                # =========================================

                try:

                    fb = feedback_engine.generate(
                        posture  = posture,
                        motion   = motion,
                        fatigue  = fatigue,
                        injuries = [],
                        snap     = snapshot,
                    )

                    if hasattr(
                        fb,
                        "primary"
                    ):

                        if fb.primary:

                            feedback = (
                                fb.primary.message
                            )

                    if getattr(
                        fb,
                        "ai_message",
                        ""
                    ):

                        feedback = fb.ai_message

                except Exception as e:

                    logger.warning(
                        "FeedbackEngine failed: %s",
                        e,
                    )

                # =========================================
                # RENDER
                # =========================================

                try:

                    rendered = renderer.render(
                        frame,
                        pose=pose,
                        snap=snapshot,
                        posture=posture,
                        motion=motion,
                    )

                    if rendered is not None:
                        frame = rendered

                except Exception as e:

                    logger.warning(
                        "Renderer failed: %s",
                        e,
                    )

            except Exception as e:

                logger.error(
                    traceback.format_exc()
                )

        # =================================================
        # FPS
        # =================================================

        current_time = time.time()

        fps = 1 / max(
            current_time - prev_time,
            1e-5,
        )

        prev_time = current_time

        # =================================================
        # UI
        # =================================================

        h, w = frame.shape[:2]

        draw_panel(
            frame,
            0,
            180,
        )

        draw_panel(
            frame,
            h - 70,
            h,
        )

        # LEFT

        draw_text(
            frame,
            f"Exercise : {prediction}",
            (20, 35),
            0.8,
            (0, 255, 0),
        )

        draw_text(
            frame,
            f"Confidence : {confidence:.2f}",
            (20, 70),
            0.65,
            (255, 255, 0),
        )

        draw_text(
            frame,
            f"Posture : {posture_label}",
            (20, 105),
            0.65,
            (0, 200, 255),
        )

        draw_text(
            frame,
            f"Posture Score : {posture_score:.0f}/100",
            (20, 140),
            0.65,
        )

        # RIGHT

        draw_text(
            frame,
            f"Reps : {reps}",
            (w - 260, 35),
            0.8,
            (0, 255, 255),
        )

        draw_text(
            frame,
            f"Fatigue : {fatigue_text}",
            (w - 260, 70),
            0.65,
            (0, 140, 255),
        )

        draw_text(
            frame,
            f"Gesture : {gesture_text}",
            (w - 260, 105),
            0.65,
            (255, 0, 255),
        )

        draw_text(
            frame,
            f"FPS : {fps:.1f}",
            (w - 260, 140),
            0.65,
        )

        # INJURY

        if (
            injury_text
            and injury_text != "Safe"
        ):

            draw_text(
                frame,
                f"WARNING : {injury_text}",
                (20, 220),
                0.75,
                (0, 0, 255),
            )

        # AI COACH

        draw_text(
            frame,
            f"AI Coach : {feedback}",
            (20, h - 25),
            0.65,
            (0, 255, 255),
        )

        # =================================================
        # WINDOW
        # =================================================

        cv2.imshow(
            WINDOW_NAME,
            frame,
        )

        # =================================================
        # KEYS
        # =================================================

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord("q"):

            logger.info(
                "Quitting..."
            )

            break

        # Reset
        elif key == ord("r"):

            try:

                if hasattr(
                    motion_analyser,
                    "reset_reps"
                ):

                    motion_analyser.reset_reps()

                elif hasattr(
                    motion_analyser,
                    "reset"
                ):

                    motion_analyser.reset()

                if hasattr(
                    fatigue_engine,
                    "reset"
                ):

                    fatigue_engine.reset()

                if hasattr(
                    injury_detector,
                    "reset"
                ):

                    injury_detector.reset()

                logger.info(
                    "Session reset"
                )

            except Exception as e:

                logger.warning(
                    "Reset failed: %s",
                    e,
                )

        # =================================================
        # FPS LIMIT
        # =================================================

        elapsed = (
            time.time() - frame_start
        )

        if elapsed < FRAME_TIME:

            time.sleep(
                FRAME_TIME - elapsed
            )

    # =====================================================
    # CLEANUP
    # =====================================================

    try:
        pose_engine.close()
    except:
        pass

    try:
        camera.release()
    except:
        pass

    cv2.destroyAllWindows()

    logger.info(
        "AI Health Trainer Closed"
    )

# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    main()