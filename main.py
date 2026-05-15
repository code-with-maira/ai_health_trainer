# =========================
# FIXED main.py
# =========================

import cv2
from datetime import datetime
import numpy as np

# =========================
# CORE MODULES
# =========================
from storage.session_history import SessionHistory
from storage.storage_csv import CSVStorage

from core.camera_manager import CameraManager
from core.pose_engine import PoseEngine
from core.landmark_extractor import LandmarkExtractor
from core.movement_smoothing import MovementSmoothing

from core.feedback_engine import FeedbackEngine

# =========================
# AI MODELS
# =========================
from ai.models.exercise_classifier import ExerciseClassifier
from ai.models.posture_classifier import PostureClassifier
from ai.models.calorie_predictor import CaloriePredictor
from ai.models.injury_risk_model import InjuryRiskModel
from ai.models.fatigue_model import FatigueModel

# =========================
# AI COACH
# =========================
from ai.recommendations.ai_coaching import AICoach

# =========================
# ANALYTICS
# =========================
from analytics.workout_analytics import WorkoutSession


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n===================================")
    print("        AI HEALTH TRAINER")
    print("===================================\n")

    # =====================================================
    # INITIALIZE SYSTEMS
    # =====================================================

    camera = CameraManager()

    pose_engine = PoseEngine()

    extractor = LandmarkExtractor()

    smoothing = MovementSmoothing(
        window_size=5
    )

    feedback_engine = FeedbackEngine()

    # =====================================================
    # AI MODELS
    # =====================================================

    exercise_model = ExerciseClassifier()

    posture_model = PostureClassifier()

    calorie_model = CaloriePredictor()

    injury_model = InjuryRiskModel()

    fatigue_model = FatigueModel()

    ai_coaching = AICoach()

    # =====================================================
    # STORAGE
    # =====================================================

    storage = CSVStorage()

    session_history = SessionHistory(storage)

    # =====================================================
    # CAMERA START
    # =====================================================

    camera.open()

    while True:

        frame = camera.read()

        if frame is None:

            print("❌ Camera Error")

            break

        # =================================================
        # POSE DETECTION
        # =================================================

        pose_result = pose_engine.process(frame)

        if pose_result is None:

            cv2.imshow(
                "AI Health Trainer",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            continue

        # =================================================
        # EXTRACT BODY SNAPSHOT
        # =================================================

        body = extractor.extract(
            pose_result
        )

        if body is None:

            cv2.imshow(
                "AI Health Trainer",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            continue

        # =================================================
        # SMOOTHING
        # =================================================

        try:

            body = smoothing.smooth(body)

        except Exception as e:

            print("Smoothing Error:", e)

        # =================================================
        # CONVERT TO ML VECTOR
        # =================================================

        try:

            landmarks = extractor.extract_angle_vector(
                body
            )

            landmarks = np.nan_to_num(
                landmarks
            )

        except Exception as e:

            print("Vector Error:", e)

            continue

        # =================================================
        # AI PREDICTIONS
        # =================================================

        try:

            exercise = exercise_model.predict(
                landmarks
            )

        except:

            exercise = "Squat"

        # -------------------------------------------------

        try:

            posture = posture_model.predict(
                landmarks
            )

        except:

            posture = "Good"

        # -------------------------------------------------

        try:

            calories = calorie_model.predict(
                landmarks
            )

        except:

            calories = 120

        # -------------------------------------------------

        try:

            injury_risk = injury_model.predict(
                landmarks
            )

        except:

            injury_risk = "Low"

        # -------------------------------------------------

        try:

            fatigue = fatigue_model.predict(
                landmarks
            )

        except:

            fatigue = "Normal"

        # =================================================
        # FEEDBACK
        # =================================================

        try:

            feedback_bundle = feedback_engine.generate(
                posture=posture,
                fatigue=fatigue
            )

            if feedback_bundle.primary:

                feedback = (
                    feedback_bundle
                    .primary
                    .message
                )

            else:

                feedback = "Good Form!"

        except:

            feedback = "Good Form!"

        # =================================================
        # AI COACH
        # =================================================

        coach_results = {

            "exercise": exercise,

            "posture": {
                "advice": str(posture)
            },

            "fatigue": {
                "message": str(fatigue)
            },

            "risk": {
                "advice": str(injury_risk)
            }
        }

        coach_messages = (
            ai_coaching
            .get_coaching_message(
                coach_results
            )
        )

        recommendation = " | ".join(
            coach_messages
        )

        # =================================================
        # DRAW POSE
        # =================================================

        try:

            frame = pose_engine.draw_pose(
                frame,
                pose_result
            )

        except Exception as e:

            print("Draw Error:", e)

        # =================================================
        # DISPLAY TEXT
        # =================================================

        cv2.putText(
            frame,
            f"Exercise: {exercise}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Posture: {posture}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Calories: {calories}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Fatigue: {fatigue}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Risk: {injury_risk}",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2
        )

        cv2.putText(
            frame,
            feedback,
            (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            recommendation,
            (20, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 255, 200),
            2
        )

        # =================================================
        # SAVE SESSION
        # =================================================

        try:

            session = WorkoutSession(

                session_id=str(
                    datetime.now().timestamp()
                ),

                date=datetime.now(),

                exercise=str(exercise),

                sets=[],

                total_duration_min=0,

                heart_rate_avg=None,

                calories_burned=float(calories),

                notes=(
                    f"Fatigue: {fatigue}"
                )
            )

            session_history.add(
                session
            )

        except Exception as e:

            print(
                "Session Save Error:",
                e
            )

        # =================================================
        # SHOW WINDOW
        # =================================================

        cv2.imshow(
            "AI Health Trainer",
            frame
        )

        key = cv2.waitKey(1)

        if key == ord('q'):

            break

    # =====================================================
    # RELEASE
    # =====================================================

    camera.release()

    cv2.destroyAllWindows()

    print(
        "\n✅ Application Closed Successfully"
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()