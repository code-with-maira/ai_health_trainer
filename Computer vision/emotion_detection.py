from deepface import DeepFace
import cv2
import numpy as np

class EmotionDetector:
    def analyze(self, frame: np.ndarray) -> list[dict]:
        try:
            results = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            return results if isinstance(results, list) else [results]
        except Exception:
            return []

    def draw(self, frame: np.ndarray, results: list[dict]) -> np.ndarray:
        for r in results:
            region = r.get("region", {})
            x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
            emotion = r.get("dominant_emotion", "")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 0), 2)
        return frame