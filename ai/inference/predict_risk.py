import numpy as np
import pickle, os

class RiskPredictor:
    def __init__(self, model_path='saved_models/injury_risk.pkl'):
        self.model = None
        self.load(model_path)

    def load(self, path):
        if os.path.exists(path):
            self.model = pickle.load(open(path, 'rb'))
            print("Risk model loaded!")
        else:
            print(f"Model nahi mila: {path}")

    def predict(self, joint_angles, speed, fatigue, form_score):
        """
        joint_angles: list of 5 angles
        speed: 0-1 (1 = fast)
        fatigue: 0-1 (1 = exhausted)
        form_score: 0-1 (1 = perfect form)
        """
        if self.model is None:
            return "Model not loaded"

        features = np.array(joint_angles + [speed, fatigue, form_score]).reshape(1, -1)
        risk = self.model.predict(features)[0]
        probs = self.model.predict_proba(features)[0]

        advice = {
            'low':    '✅ Form acha hai — continue!',
            'medium': '⚠️ Thoda slow down karo',
            'high':   '🛑 Ruko! Form theek karo'
        }

        color = {'low': 'green', 'medium': 'orange', 'high': 'red'}

        return {
            'risk':       risk,
            'confidence': round(max(probs) * 100, 1),
            'advice':     advice.get(risk, ''),
            'color':      color.get(risk, 'white')
        }


if __name__ == "__main__":
    pred = RiskPredictor()
    result = pred.predict(
        joint_angles=[0.5, 0.3, 0.8, 0.6, 0.4],
        speed=0.7,
        fatigue=0.3,
        form_score=0.8
    )
    print("Risk:", result)