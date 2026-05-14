# =========================================================
# realtime_inference.py
# FIXED REALTIME AI HEALTH TRAINER
# =========================================================

import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


class RealtimeInference:

    def __init__(self, models_dir='saved_models'):

        # ============================================
        # MediaPipe
        # ============================================

        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ============================================
        # Models
        # ============================================

        self.models_dir = models_dir
        self.models = {}

        self.load_models()

    # =====================================================
    # LOAD MODELS
    # =====================================================

    def load_models(self):

        model_files = {
            'exercise': 'exercise_classifier.pkl',
        }

        for name, fname in model_files.items():

            path = os.path.join(self.models_dir, fname)

            if os.path.exists(path):

                with open(path, 'rb') as f:
                    self.models[name] = pickle.load(f)

                print(f"✅ Loaded: {name}")

            else:
                print(f"❌ Not found: {name}")

    # =====================================================
    # CALCULATE ANGLE
    # =====================================================

    def calc_angle(self, a, b, c):

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = (
            np.arctan2(c[1] - b[1], c[0] - b[0]) -
            np.arctan2(a[1] - b[1], a[0] - b[0])
        )

        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180:
            angle = 360 - angle

        return angle

    # =====================================================
    # EXTRACT 10 ANGLE FEATURES
    # =====================================================

    def extract_features(self, results):

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark

        # ============================================
        # LEFT SIDE LANDMARKS
        # ============================================

        shoulder = [
            landmarks[11].x,
            landmarks[11].y
        ]

        elbow = [
            landmarks[13].x,
            landmarks[13].y
        ]

        wrist = [
            landmarks[15].x,
            landmarks[15].y
        ]

        hip = [
            landmarks[23].x,
            landmarks[23].y
        ]

        knee = [
            landmarks[25].x,
            landmarks[25].y
        ]

        ankle = [
            landmarks[27].x,
            landmarks[27].y
        ]

        # ============================================
        # ANGLES
        # ============================================

        shoulder_angle = self.calc_angle(
            elbow,
            shoulder,
            hip
        )

        elbow_angle = self.calc_angle(
            shoulder,
            elbow,
            wrist
        )

        hip_angle = self.calc_angle(
            shoulder,
            hip,
            knee
        )

        knee_angle = self.calc_angle(
            hip,
            knee,
            ankle
        )

        ankle_angle = self.calc_angle(
            knee,
            ankle,
            [ankle[0], ankle[1] + 0.1]
        )

        # ============================================
        # GROUND ANGLES
        # ============================================

        shoulder_ground = shoulder_angle
        elbow_ground = elbow_angle
        hip_ground = hip_angle
        knee_ground = knee_angle
        ankle_ground = ankle_angle

        # ============================================
        # FINAL FEATURES
        # ============================================

        features = np.array([
            shoulder_angle,
            elbow_angle,
            hip_angle,
            knee_angle,
            ankle_angle,
            shoulder_ground,
            elbow_ground,
            hip_ground,
            knee_ground,
            ankle_ground
        ], dtype=np.float32)

        return features.reshape(1, -1)

    # =====================================================
    # DRAW INFO
    # =====================================================

    def draw_info(self, frame, predictions):

        y = 30

        for key, val in predictions.items():

            cv2.putText(
                frame,
                f"{key}: {val}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            y += 35

        return frame

    # =====================================================
    # RUN CAMERA
    # =====================================================

    def run(self):

        cap = cv2.VideoCapture(0)

        print("\n✅ Camera Started")
        print("Press 'q' to quit\n")

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            # ============================================
            # RGB
            # ============================================

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = self.pose.process(rgb)

            predictions = {}

            # ============================================
            # FEATURE EXTRACTION
            # ============================================

            features = self.extract_features(results)

            # ============================================
            # EXERCISE PREDICTION
            # ============================================

            if features is not None:

                try:

                    if 'exercise' in self.models:

                        model_data = self.models['exercise']

                        # Handle both styles
                        if isinstance(model_data, dict):
                            model = model_data["model"]
                            encoder = model_data["label_encoder"]

                            pred = model.predict(features)[0]

                            label = encoder.inverse_transform([pred])[0]

                        else:
                            label = model_data.predict(features)[0]

                        predictions['Exercise'] = label

                except Exception as e:
                    predictions['Error'] = str(e)

            # ============================================
            # DRAW LANDMARKS
            # ============================================

            if results.pose_landmarks:

                self.mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )

            # ============================================
            # DRAW TEXT
            # ============================================

            frame = self.draw_info(
                frame,
                predictions
            )

            cv2.imshow(
                'AI Health Trainer',
                frame
            )

            # ============================================
            # EXIT
            # ============================================

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # ============================================
        # RELEASE
        # ============================================

        cap.release()
        cv2.destroyAllWindows()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    inference = RealtimeInference()

    inference.run()