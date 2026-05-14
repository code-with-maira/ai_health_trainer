import numpy as np
import cv2

class Augmentor:
    def __init__(self, seed=42):
        np.random.seed(seed)

    # ─────────────────────────────
    # IMAGE AUGMENTATION
    # ─────────────────────────────

    def flip_image(self, img):
        return cv2.flip(img, 1)

    def rotate_image(self, img, angle=None):
        if angle is None:
            angle = np.random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))

    def add_noise_image(self, img, amount=0.02):
        noise = np.random.normal(0, amount * 255, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy

    def adjust_brightness(self, img, factor=None):
        if factor is None:
            factor = np.random.uniform(0.7, 1.3)
        return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def augment_image(self, img, n=4):
        augmented = [img]
        ops = [
            self.flip_image,
            self.rotate_image,
            self.add_noise_image,
            self.adjust_brightness
        ]

        for op in ops[:n]:
            try:
                augmented.append(op(img.copy()))
            except Exception as e:
                print(f"Augmentation skip: {e}")

        return augmented

    # ─────────────────────────────
    # ANGLE / TABULAR AUGMENTATION (FIXED)
    # ─────────────────────────────

    def augment_keypoints(self, X, y, multiplier=2):
        """
        Angle-based dataset augmentation (NO reshape, NO keypoints)
        X: (n_samples, n_features)
        y: (n_samples,)
        """

        X_aug = [X]
        y_aug = [y]

        for _ in range(multiplier - 1):
            noise = np.random.normal(0, 0.5, X.shape)  # small noise
            X_new = X + noise

            X_aug.append(X_new)
            y_aug.append(y)

        X_final = np.vstack(X_aug)
        y_final = np.hstack(y_aug)

        print(f"Dataset: {len(X)} → {len(X_final)} samples (x{multiplier})")

        return X_final, y_final


# ─────────────────────────────
# TEST
# ─────────────────────────────
if __name__ == "__main__":
    aug = Augmentor()

    dummy_img = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
    imgs = aug.augment_image(dummy_img, n=4)
    print(f"Image augmentation: {len(imgs)} images")

    X_dummy = np.random.rand(100, 11)  # angle dataset (11 features)
    y_dummy = np.random.randint(0, 5, 100)

    X_aug, y_aug = aug.augment_keypoints(X_dummy, y_dummy, multiplier=3)
    print(f"Final shape: {X_aug.shape}")