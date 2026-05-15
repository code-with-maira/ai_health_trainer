import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, max_hands=2, detection_conf=0.7, tracking_conf=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame: np.ndarray) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def draw(self, frame: np.ndarray, results) -> np.ndarray:
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def get_landmarks(self, results, frame_shape: tuple) -> list[list[tuple]]:
        all_hands = []
        h, w = frame_shape[:2]
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]
                all_hands.append(points)
        return all_hands