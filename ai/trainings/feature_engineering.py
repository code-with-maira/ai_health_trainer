import numpy as np
import pandas as pd

class FeatureEngineer:
    def __init__(self):
        # MediaPipe ke 33 body landmarks ke indices
        self.joints = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13,    'right_elbow': 14,
            'left_wrist': 15,    'right_wrist': 16,
            'left_hip': 23,      'right_hip': 24,
            'left_knee': 25,     'right_knee': 26,
            'left_ankle': 27,    'right_ankle': 28
        }

    def calculate_angle(self, a, b, c):
        """3 points ke darmiyan angle nikalo (degrees mein)"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def calculate_distance(self, p1, p2):
        """Do points ke darmiyan distance"""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def extract_angles(self, keypoints):
        """
        keypoints: list of [x, y] for each joint
        Returns: list of important body angles
        """
        kp = keypoints  # shortcut

        angles = []
        try:
            # Right arm angle (shoulder-elbow-wrist)
            angles.append(self.calculate_angle(kp[12], kp[14], kp[16]))
            # Left arm angle
            angles.append(self.calculate_angle(kp[11], kp[13], kp[15]))
            # Right leg angle (hip-knee-ankle)
            angles.append(self.calculate_angle(kp[24], kp[26], kp[28]))
            # Left leg angle
            angles.append(self.calculate_angle(kp[23], kp[25], kp[27]))
            # Torso angle (shoulder-hip-knee)
            angles.append(self.calculate_angle(kp[12], kp[24], kp[26]))
            angles.append(self.calculate_angle(kp[11], kp[23], kp[25]))
        except (IndexError, Exception) as e:
            print(f"Angle error: {e}")
            angles = [0.0] * 6

        return angles

    def extract_distances(self, keypoints):
        """Key distances between joints"""
        kp = keypoints
        distances = []
        try:
            # Shoulder width
            distances.append(self.calculate_distance(kp[11], kp[12]))
            # Hip width
            distances.append(self.calculate_distance(kp[23], kp[24]))
            # Left arm length
            distances.append(self.calculate_distance(kp[11], kp[15]))
            # Right arm length
            distances.append(self.calculate_distance(kp[12], kp[16]))
            # Torso height
            distances.append(self.calculate_distance(kp[11], kp[23]))
        except (IndexError, Exception) as e:
            print(f"Distance error: {e}")
            distances = [0.0] * 5

        return distances

    def extract_all_features(self, keypoints):
        """Ek frame se sab features nikalo"""
        angles = self.extract_angles(keypoints)
        distances = self.extract_distances(keypoints)
        # Flat list of keypoint coordinates bhi add karo
        flat_kp = np.array(keypoints).flatten().tolist()
        return angles + distances + flat_kp

    def process_dataset(self, df, keypoint_cols, label_col):
        """Poora dataset process karo"""
        features_list = []
        for _, row in df.iterrows():
            kps = row[keypoint_cols].values.reshape(-1, 2).tolist()
            features = self.extract_all_features(kps)
            features_list.append(features)
        X = np.array(features_list)
        y = df[label_col].values
        print(f"Features extracted: {X.shape}")
        return X, y


# Test karo VS Code mein
if __name__ == "__main__":
    fe = FeatureEngineer()

    # Ek frame test (33 keypoints, har ek ka x,y)
    dummy_keypoints = [[np.random.rand(), np.random.rand()] for _ in range(33)]

    angles = fe.extract_angles(dummy_keypoints)
    print(f"Angles ({len(angles)}): {[round(a,1) for a in angles]}")

    distances = fe.extract_distances(dummy_keypoints)
    print(f"Distances ({len(distances)}): {[round(d,3) for d in distances]}")

    all_features = fe.extract_all_features(dummy_keypoints)
    print(f"Total features: {len(all_features)}")
    print("Feature engineering working!")