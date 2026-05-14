import numpy as np
import cv2

class Augmentor:
    def __init__(self, seed=42):
        np.random.seed(seed)

    # ─── IMAGE AUGMENTATION (Emotion classifier ke liye) ───

    def flip_image(self, img):
        """Image ko horizontally paltao"""
        return cv2.flip(img, 1)

    def rotate_image(self, img, angle=None):
        """Image ko thoda rotate karo"""
        if angle is None:
            angle = np.random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))

    def add_noise(self, img, amount=0.02):
        """Thoda noise add karo overfitting rokne ke liye"""
        noise = np.random.normal(0, amount * 255, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy

    def adjust_brightness(self, img, factor=None):
        """Brightness change karo"""
        if factor is None:
            factor = np.random.uniform(0.7, 1.3)
        return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def augment_image(self, img, n=4):
        """Ek image se n augmented versions banao"""
        augmented = [img]
        ops = [self.flip_image, self.rotate_image,
               self.add_noise, self.adjust_brightness]
        for op in ops[:n]:
            try:
                augmented.append(op(img.copy()))
            except Exception as e:
                print(f"Augmentation skip: {e}")
        return augmented

    # ─── KEYPOINT AUGMENTATION (Exercise/Posture ke liye) ───

    def flip_keypoints(self, keypoints, img_width=1.0):
        """Keypoints ko mirror karo"""
        flipped = keypoints.copy()
        flipped[:, 0] = img_width - flipped[:, 0]
        return flipped

    def add_keypoint_noise(self, keypoints, std=0.01):
        """Keypoints mein thoda noise add karo"""
        noise = np.random.normal(0, std, keypoints.shape)
        return keypoints + noise

    def scale_keypoints(self, keypoints, factor=None):
        """Keypoints ko scale karo (zoom in/out simulate)"""
        if factor is None:
            factor = np.random.uniform(0.85, 1.15)
        center = keypoints.mean(axis=0)
        return (keypoints - center) * factor + center

    def augment_keypoints(self, X, y, multiplier=3):
        """
        Keypoint dataset ko multiplier guna bada karo
        X: (n_samples, n_features)
        y: (n_samples,)
        """
        X_aug, y_aug = [X], [y]

        for _ in range(multiplier - 1):
            X_new = []
            for sample in X:
                kps = sample.reshape(-1, 2)
                # Random augmentation choose karo
                choice = np.random.randint(3)
                if choice == 0:
                    kps = self.flip_keypoints(kps)
                elif choice == 1:
                    kps = self.add_keypoint_noise(kps)
                else:
                    kps = self.scale_keypoints(kps)
                X_new.append(kps.flatten())
            X_aug.append(np.array(X_new))
            y_aug.append(y)

        X_final = np.vstack(X_aug)
        y_final = np.concatenate(y_aug)
        print(f"Dataset: {len(X)} → {len(X_final)} samples (x{multiplier})")
        return X_final, y_final


# Test karo VS Code mein
if __name__ == "__main__":
    aug = Augmentor()

    # Image augmentation test
    dummy_img = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
    imgs = aug.augment_image(dummy_img, n=4)
    print(f"Image augmentation: {len(imgs)} images bani")

    # Keypoint augmentation test
    X_dummy = np.random.rand(100, 66)  # 33 keypoints x 2
    y_dummy = np.array(['squat','pushup','plank'] * 33 + ['squat'])
    X_aug, y_aug = aug.augment_keypoints(X_dummy, y_dummy, multiplier=3)
    print(f"Keypoint augmentation: {X_aug.shape}")
    print("Augmentation working!")