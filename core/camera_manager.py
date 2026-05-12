"""
camera_manager.py
-----------------
Advanced camera management for AI fitness systems.

Features:
- Camera open validation
- Configurable resolution + FPS
- Context manager support
- Safe release
- Multiple camera index support
- Frame read retry logic
- Camera properties logging
"""

import cv2
import logging

logger = logging.getLogger(__name__)


class CameraManager:

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        max_retries: int = 3,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.max_retries = max_retries

        self.cap: cv2.VideoCapture = None

    # ================================================================= #

    def open(self):

        self.cap = cv2.VideoCapture(self.camera_index)

        # FIX 1: Validate camera opened successfully
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Camera index {self.camera_index} open nahi ho saki. "
                f"Webcam connected hai? Doosra index try karo."
            )

        # FIX 2: Set resolution and FPS explicitly
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS,          self.fps)

        # Log actual properties (camera may not support requested values)
        actual_w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            "Camera %d opened — resolution: %dx%d @ %.1f fps",
            self.camera_index,
            actual_w,
            actual_h,
            actual_fps,
        )

        return self

    # ================================================================= #

    def read(self):
        """
        Read a frame with retry logic.
        Returns flipped frame or None if all retries fail.
        """

        if self.cap is None or not self.cap.isOpened():
            logger.warning("read() called but camera is not open")
            return None

        for attempt in range(self.max_retries):

            success, frame = self.cap.read()

            if success and frame is not None:
                return cv2.flip(frame, 1)

            logger.warning(
                "Frame read failed — attempt %d/%d",
                attempt + 1,
                self.max_retries,
            )

        logger.error(
            "Camera %d: frame read failed after %d retries",
            self.camera_index,
            self.max_retries,
        )

        return None

    # ================================================================= #

    def release(self):

        # FIX 3: Safe release — check before releasing
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera %d released", self.camera_index)

        cv2.destroyAllWindows()

        self.cap = None

    # ================================================================= #

    #: Context manager support
    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    # ================================================================= #

    @property
    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    # ================================================================= #

    @staticmethod
    def list_cameras(max_check: int = 5) -> list:
        """
        Returns list of available camera indices.
        Useful for selecting external webcams.
        """
        available = []

        for i in range(max_check):

            cap = cv2.VideoCapture(i)

            if cap.isOpened():
                available.append(i)
                cap.release()

        logger.info("Available cameras: %s", available)

        return available