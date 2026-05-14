import numpy as np
import pickle, os

class FatiguePredictor:
    def __init__(self, model_path='saved_models/fatigue_model.pkl'):
        self.model = None
        self.load(model_path)
        self.movement_buffer = []
        self.buffer_size = 30  # 30 frames

    def load(self, path):
        if os.path.exists(path):
            self.model = pickle.load(open(path, 'rb'))
            print("Fatigue model loaded!")
        else:
            print(f"Model nahi mila: {path}")

    def add_frame(self, keypoints):
        """Har frame ke keypoints add karo buffer mein"""
        self.movement_buffer.append(keypoints)
        if len(self.movement_buffer) > self.buffer_size:
            self.movement_buffer.pop(0)

    def extract_features(self):
        """Buffer se fatigue features nikalo"""
        if len(self.movement_buffer) < 5:
            return None

        data = np.array(self.movement_buffer)
        speeds = np.diff(data, axis=0)

        features = [
            np.mean(np.abs(speeds)),
            np.std(speeds),
            np.max(data) - np.min(data),
            np.mean(np.abs(np.diff(speeds, axis=0)))
        ]
        return np.array(features).reshape(1, -1)

    def predict(self):
        """Current fatigue level predict karo"""
        if self.model is None:
            return "Model not loaded"

        features = self.extract_features()
        if features is None:
            return "Zyada data chahiye"

        level = self.model.predict(features)[0]

        messages = {
            'fresh':        '💪 Energy full hai — keep going!',
            'mild_fatigue': '😊 Thoda thaka ho — pace thodi kam karo',
            'high_fatigue': '😓 Zyada thaka ho — rest lo',
            'exhausted':    '🛑 Ruko! Body rest maang rahi hai'
        }

        return {
            'fatigue_level': level,
            'message':       messages.get(level, '')
        }


if __name__ == "__main__":
    pred = FatiguePredictor()
    # Simulate frames
    for i in range(35):
        dummy_kp = np.random.rand(10)
        pred.add_frame(dummy_kp)
    result = pred.predict()
    print("Fatigue:", result)