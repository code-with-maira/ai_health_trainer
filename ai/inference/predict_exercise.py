import numpy as np
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ExercisePredictor:
    def __init__(self, model_path='saved_models/exercise_classifier.pkl'):
        self.model = None
        self.scaler = None
        self.encoder = None
        self.load(model_path)

    def load(self, path):
        """Load trained model"""
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)

                if isinstance(data, dict):
                    self.model = data.get('model')
                    self.scaler = data.get('scaler')
                    self.encoder = data.get('encoder')
                else:
                    self.model = data

                print(" Exercise model loaded!")

            except Exception as e:
                print(f" Model load error: {e}")

        else:
            print(f" Model nahi mila: {path} — pehle train karo")

    def predict(self, keypoints):
        """
        keypoints: numpy array shape (66,)
        Returns: predicted exercise name
        """

        if self.model is None:
            return "Model not loaded"

        try:
            kp = np.array(keypoints, dtype=np.float32).reshape(1, -1)

            # Safety check
            if kp.shape[1] != 66:
                return f"Invalid input shape: {kp.shape}"

            if self.scaler is not None:
                kp = self.scaler.transform(kp)

            pred = self.model.predict(kp)[0]

            if self.encoder is not None:
                pred = self.encoder.inverse_transform([pred])[0]

            return pred

        except Exception as e:
            return f"Prediction error: {e}"

    def predict_proba(self, keypoints):
        """Return probability of each exercise"""

        if self.model is None:
            return {}

        try:
            kp = np.array(keypoints, dtype=np.float32).reshape(1, -1)

            if self.scaler is not None:
                kp = self.scaler.transform(kp)

            probs = self.model.predict_proba(kp)[0]

            # Get class labels
            if self.encoder is not None:
                classes = self.encoder.inverse_transform(self.model.classes_)
            else:
                classes = self.model.classes_

            return {
                str(cls): round(float(prob), 3)
                for cls, prob in zip(classes, probs)
            }

        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":

    predictor = ExercisePredictor()

    # Dummy test
    dummy_keypoints = np.random.rand(66)

    result = predictor.predict(dummy_keypoints)
    print(f"\n Predicted Exercise: {result}")

    probs = predictor.predict_proba(dummy_keypoints)
    print("\n Probabilities:")
    print(probs)