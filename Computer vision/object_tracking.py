import cv2
import numpy as np

class ObjectTracker:
    TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "mosse": cv2.legacy.TrackerMOSSE_create,
    }

    def __init__(self, tracker_type="csrt"):
        self.tracker = self.TRACKERS.get(tracker_type, cv2.TrackerCSRT_create)()
        self.initialized = False

    def init(self, frame: np.ndarray, bbox: tuple):
        """bbox = (x, y, w, h)"""
        self.tracker.init(frame, bbox)
        self.initialized = True

    def update(self, frame: np.ndarray) -> tuple[bool, tuple]:
        if not self.initialized:
            return False, ()
        success, box = self.tracker.update(frame)
        return success, box

    def draw(self, frame: np.ndarray, box: tuple, success: bool) -> np.ndarray:
        if success:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)
            cv2.putText(frame, "Tracking", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        else:
            cv2.putText(frame, "Lost", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame