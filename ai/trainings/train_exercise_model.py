import numpy as np
import pandas as pd
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.trainings.preprocessing import Preprocessor
from ai.trainings.feature_engineering import FeatureEngineer
from ai.trainings.augmentation import Augmentor
from ai.models.exercise_classifier import ExerciseClassifier
from sklearn.metrics import classification_report, accuracy_score

def train(data_path='data/raw/exercise_dataset.csv'):
    print("=" * 60)
    print("        EXERCISE MODEL TRAINING STARTED")
    print("=" * 60)

    # ─────────────────────────────
    # 1. LOAD + CLEAN
    # ─────────────────────────────
    print("\n[1/5] Loading data...")
    prep = Preprocessor()
    df = prep.load_data(data_path)
    df = prep.clean_data(df)

    # ─────────────────────────────
    # 2. FEATURE SELECTION
    # ─────────────────────────────
    print("\n[2/5] Feature extraction...")

    feature_cols = [c for c in df.columns if c not in ['Label', 'Side']]
    label_col = 'Label'

    X = df[feature_cols].values.astype(np.float32)
    y = prep.encode_labels(df[label_col].values)

    print(f"Features used: {len(feature_cols)}")
    print(f"Classes: {list(prep.label_encoder.classes_)}")

    # ─────────────────────────────
    # 3. AUGMENTATION
    # ─────────────────────────────
    print("\n[3/5] Data augmentation...")

    aug = Augmentor()
    X, y = aug.augment_keypoints(X, y, multiplier=3)

    # ─────────────────────────────
    # 4. SPLIT + NORMALIZE
    # ─────────────────────────────
    print("\n[4/5] Train/Val/Test split...")

    X_train, X_val, X_test, y_train, y_val, y_test = prep.split(X, y)

    X_train, X_test = prep.normalize(X_train, X_test)
    X_val = prep.scaler.transform(X_val)

    # ─────────────────────────────
    # 5. MODEL TRAINING
    # ─────────────────────────────
    print("\n[5/5] Training model...")

    clf = ExerciseClassifier()
    clf.train(X_train, y_train)

    # ─────────────────────────────
    # EVALUATION (FIXED + SAFE)
    # ─────────────────────────────
    print("\n" + "=" * 60)
    print("           MODEL EVALUATION")
    print("=" * 60)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc * 100:.2f}%\n")

    print(classification_report(
        y_test,
        y_pred,
        labels=list(range(len(prep.label_encoder.classes_))),
        target_names=[str(c) for c in prep.label_encoder.classes_]
    ))

    # ─────────────────────────────
    # SAVE MODEL
    # ─────────────────────────────
    print("\nSaving models...")

    clf.save('saved_models/exercise_classifier.pkl')
    prep.save('saved_models/preprocessor.pkl')

    print("\nTraining Completed Successfully 🚀")

    return clf, prep


if __name__ == "__main__":
    clf, prep = train(
        data_path='data/raw/exercise_dataset.csv'
    )