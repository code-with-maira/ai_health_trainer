import numpy as np
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PosturePredictor:

    def __init__(self, model_path='saved_models/posture_clf.pkl'):
        self.model = None
        self.scaler = None
        self.load(model_path)

    def load(self, path):

        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)

                self.model = data.get('model')
                self.scaler = data.get('scaler')

                print("✅ Posture model loaded!")

            except Exception as e:
                print(f"❌ Model loading error: {e}")

        else:
            print(f"❌ Model nahi mila: {path}")

    def calculate_angles(self, keypoints):
        """
        Calculate body angles from keypoints
        keypoints shape: (17,2)
        """

        angles = []

        kp = np.array(keypoints, dtype=np.float32).reshape(-1, 2)

        for i in range(len(kp) - 2):

            a = kp[i]
            b = kp[i + 1]
            c = kp[i + 2]

            ba = a - b
            bc = c - b

            denominator = (
                np.linalg.norm(ba) *
                np.linalg.norm(bc)
            ) + 1e-8

            cosine = np.dot(ba, bc) / denominator

            angle = np.degrees(
                np.arccos(np.clip(cosine, -1.0, 1.0))
            )

            angles.append(angle)

        return np.array(angles, dtype=np.float32)

    def predict(self, keypoints):

        if self.model is None:
            return {
                "error": "Model not loaded"
            }

        try:

            kp = np.array(keypoints)

            # Safety check
            if kp.shape != (17, 2):
                return {
                    "error": f"Invalid shape: {kp.shape}"
                }

            angles = self.calculate_angles(kp)

            # Ensure fixed feature size
            angles = angles[:15]

            if len(angles) < 15:
                angles = np.pad(
                    angles,
                    (0, 15 - len(angles)),
                    mode='constant'
                )

            angles = angles.reshape(1, -1)

            if self.scaler is not None:
                angles = self.scaler.transform(angles)

            label = self.model.predict(angles)[0]

            # Probability support check
            if hasattr(self.model, "predict_proba"):
                prob = float(
                    np.max(
                        self.model.predict_proba(angles)[0]
                    )
                )
            else:
                prob = 1.0

            advice = {
                'good_posture': '✅ Shabaash! Posture acha hai',
                'bad_posture': '⚠️ Posture theek karo — kamar seedhi rakho',
                'slouching': '⚠️ Jhukna band karo — shoulders peeche karo',
                'leaning': '⚠️ Seedha kharo — ek taraf mat jhuko'
            }

            return {
                'posture': str(label),
                'confidence': round(prob * 100, 1),
                'advice': advice.get(label, '')
            }

        except Exception as e:
            return {
                "error": str(e)
            }


if __name__ == "__main__":

    pred = PosturePredictor()

    dummy_kp = [
        [np.random.rand(), np.random.rand()]
        for _ in range(17)
    ]

    result = pred.predict(dummy_kp)

    print("\n Result:")
    print(result)