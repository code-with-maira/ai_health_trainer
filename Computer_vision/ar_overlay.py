import cv2
import numpy as np

class AROverlay:
    def overlay_image(self, background: np.ndarray, overlay: np.ndarray,
                      position: tuple, scale: float = 1.0) -> np.ndarray:
        x, y = position
        ov = cv2.resize(overlay, None, fx=scale, fy=scale)
        h, w = ov.shape[:2]

        if ov.shape[2] == 4:  # Has alpha channel
            alpha = ov[:, :, 3] / 255.0
            ov_bgr = ov[:, :, :3]
        else:
            alpha = np.ones((h, w))
            ov_bgr = ov

        y1, y2 = max(0, y), min(background.shape[0], y + h)
        x1, x2 = max(0, x), min(background.shape[1], x + w)

        for c in range(3):
            background[y1:y2, x1:x2, c] = (
                alpha[:y2-y1, :x2-x1] * ov_bgr[:y2-y1, :x2-x1, c] +
                (1 - alpha[:y2-y1, :x2-x1]) * background[y1:y2, x1:x2, c]
            )
        return background

    def draw_3d_box(self, frame: np.ndarray, points_3d: np.ndarray,
                    rvec, tvec, camera_matrix, dist_coeffs) -> np.ndarray:
        pts, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
        pts = pts.reshape(-1, 2).astype(int)
        for i in range(4):
            cv2.line(frame, pts[i], pts[(i+1) % 4], (0, 255, 0), 2)
            cv2.line(frame, pts[i+4], pts[(i+1) % 4 + 4], (0, 255, 0), 2)
            cv2.line(frame, pts[i], pts[i+4], (255, 0, 0), 2)
        return frame