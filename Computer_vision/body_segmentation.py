import cv2
import mediapipe as mp
import numpy as np

class BodySegmenter:
    def __init__(self):
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)

    def segment(self, frame: np.ndarray, bg_color=(0, 0, 0)) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.segmenter.process(rgb)
        mask = result.segmentation_mask > 0.5
        bg = np.full_like(frame, bg_color, dtype=np.uint8)
        output = np.where(mask[:, :, None], frame, bg)
        return output

    def blur_background(self, frame: np.ndarray, blur_amount=55) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.segmenter.process(rgb)
        mask = result.segmentation_mask > 0.5
        blurred = cv2.GaussianBlur(frame, (blur_amount, blur_amount), 0)
        output = np.where(mask[:, :, None], frame, blurred)
        return output