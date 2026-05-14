import sys
import os
import numpy as np
from collections import Counter

# Parent directory add
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# Import predictors
from ai.inference.predict_exercise import ExercisePredictor
from ai.inference.predict_posture import PosturePredictor
from ai.inference.predict_fatigue import FatiguePredictor
from ai.inference.predict_risk import RiskPredictor


class AICoach:

    def __init__(self,
                 models_dir='saved_models'):

        print("🤖 AI Coach load ho raha hai...")

        self.exercise_pred = ExercisePredictor(
            f'{models_dir}/exercise_classifier.pkl'
        )

        self.posture_pred = PosturePredictor(
            f'{models_dir}/posture_clf.pkl'
        )

        self.fatigue_pred = FatiguePredictor(
            f'{models_dir}/fatigue_model.pkl'
        )

        self.risk_pred = RiskPredictor(
            f'{models_dir}/injury_risk.pkl'
        )

        # Session history
        self.session_data = []

        print("✅ AI Coach ready!")

    # =====================================================
    # FRAME ANALYSIS
    # =====================================================
    def analyze_frame(self,
                      keypoints,
                      speed=0.5):

        results = {}

        try:

            # Convert to numpy
            keypoints = np.array(
                keypoints,
                dtype=np.float32
            )

            # Safety check
            if len(keypoints) < 34:
                return {
                    "error":
                    "Insufficient keypoints"
                }

            # =========================================
            # Exercise Prediction
            # =========================================
            try:

                exercise = self.exercise_pred.predict(
                    keypoints
                )

                results['exercise'] = exercise

            except Exception as e:

                results['exercise'] = 'Unknown'

                print(
                    f"Exercise Error: {e}"
                )

            # =========================================
            # Posture Prediction
            # =========================================
            try:

                kp_2d = [
                    [keypoints[i],
                     keypoints[i + 1]]

                    for i in range(
                        0,
                        min(34, len(keypoints)),
                        2
                    )
                ]

                posture = self.posture_pred.predict(
                    kp_2d
                )

                results['posture'] = posture

            except Exception as e:

                results['posture'] = {
                    'posture': 'Unknown'
                }

                print(
                    f"Posture Error: {e}"
                )

            # =========================================
            # Fatigue Prediction
            # =========================================
            try:

                self.fatigue_pred.add_frame(
                    keypoints[:10]
                )

                fatigue = self.fatigue_pred.predict()

                results['fatigue'] = fatigue

            except Exception as e:

                results['fatigue'] = {
                    'fatigue_level': 'Unknown'
                }

                print(
                    f"Fatigue Error: {e}"
                )

            # =========================================
            # Injury Risk Prediction
            # =========================================
            try:

                angles = keypoints[:5].tolist()

                fatigue_val = 0.3

                risk = self.risk_pred.predict(
                    angles=angles,
                    speed=speed,
                    fatigue=fatigue_val,
                    stability=0.8
                )

                results['risk'] = risk

            except Exception as e:

                results['risk'] = {
                    'risk': 'Unknown'
                }

                print(
                    f"Risk Error: {e}"
                )

            # =========================================
            # Timestamp
            # =========================================
            results['frame_speed'] = speed

            # Save session history
            self.session_data.append(
                results
            )

            # Prevent memory overflow
            if len(self.session_data) > 5000:
                self.session_data.pop(0)

            return results

        except Exception as e:

            return {
                "error": str(e)
            }

    # =====================================================
    # COACHING MESSAGE
    # =====================================================
    def get_coaching_message(
            self,
            results):

        messages = []

        # -----------------------------------------
        # Posture advice
        # -----------------------------------------
        posture = results.get(
            'posture',
            {}
        )

        if isinstance(posture, dict):

            advice = posture.get(
                'advice',
                ''
            )

            if advice:
                messages.append(advice)

        # -----------------------------------------
        # Fatigue advice
        # -----------------------------------------
        fatigue = results.get(
            'fatigue',
            {}
        )

        if isinstance(fatigue, dict):

            msg = fatigue.get(
                'message',
                ''
            )

            if msg:
                messages.append(msg)

        # -----------------------------------------
        # Risk advice
        # -----------------------------------------
        risk = results.get(
            'risk',
            {}
        )

        if isinstance(risk, dict):

            advice = risk.get(
                'advice',
                ''
            )

            if advice:
                messages.append(advice)

        # Default message
        if not messages:
            messages.append(
                "✅ Workout acha chal raha hai"
            )

        return messages

    # =====================================================
    # SESSION SUMMARY
    # =====================================================
    def get_session_summary(self):

        if not self.session_data:

            return {
                "message":
                "Koi session data nahi"
            }

        exercises = [

            d.get('exercise')

            for d in self.session_data

            if d.get('exercise')
            not in [None, 'Unknown']
        ]

        top_exercises = Counter(
            exercises
        ).most_common(3)

        duration_secs = round(
            len(self.session_data) / 30,
            1
        )

        return {

            'total_frames':
                len(self.session_data),

            'duration_secs':
                duration_secs,

            'top_exercises':
                top_exercises
        }


# =========================================================
# TESTING
# =========================================================
if __name__ == "__main__":

    coach = AICoach()

    print("\n🎥 Testing AI Coach...\n")

    for i in range(10):

        # Dummy 66 keypoints
        kp = np.random.rand(66)

        results = coach.analyze_frame(
            kp
        )

        messages = coach.get_coaching_message(
            results
        )

        print(f"\nFrame {i+1}")
        print("Results:", results)
        print("Coach:", messages)

    summary = coach.get_session_summary()

    print("\n SESSION SUMMARY")
    print(summary)