import numpy as np
import pandas as pd
import sys, os

# Path set karo taake baaki files import hon
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.trainings.preprocessing import Preprocessor
from ai.trainings.feature_engineering import FeatureEngineer
from ai.trainings.augmentation import Augmentor
from ai.models.exercise_classifier import ExerciseClassifier
from sklearn.metrics import classification_report

def train(data_path='data/raw/exercise_dataset.csv'):
    print("=" * 50)
    print("Exercise Model Training Shuru")
    print("=" * 50)

    # Step 1: Data load aur clean karo
    print("\n[1/5] Data load ho raha hai...")
    prep = Preprocessor()
    df = prep.load_data(data_path)
    df = prep.clean_data(df)

    # Step 2: Features nikalo
    print("\n[2/5] Features extract ho rahi hain...")
    fe = FeatureEngineer()

    # Dataset ke hisaab se columns change karo
    feature_cols = [c for c in df.columns if c != 'Label','Side']
    label_col = 'Label'

    X = df[feature_cols].values
    y = prep.encode_labels(df[label_col].values)

    # Step 3: Augmentation
    print("\n[3/5] Data augmentation...")
    aug = Augmentor()
    X, y = aug.augment_keypoints(X, y, multiplier=3)  
    # Step 4: Split aur normalize
    print("\n[4/5] Train/val/test split...")
    X_train, X_val, X_test, y_train, y_val, y_test = prep.split(X, y)
    X_train, X_test = prep.normalize(X_train, X_test)
    X_val = prep.scaler.transform(X_val)

    # Step 5: Model train karo
    print("\n[5/5] Model train ho raha hai...")
    clf = ExerciseClassifier()
    clf.train(X_train, y_train)

    # Results
    print("\n--- Results ---")
    y_pred = clf.predict(X_test)
    print(classification_report(
        y_test, y_pred,
        target_names=prep.label_encoder.classes_
    ))

    # Save karo
    print("\nModels save ho rahe hain...")
    clf.save('saved_models/exercise_classifier.pkl')
    prep.save('saved_models/preprocessor.pkl')

    print("\nTraining Complete!")
    return clf, prep


if __name__ == "__main__":
    # VS Code mein: F5 dabao ya terminal mein:
    # python ai/training/train_exercise_model.py

    # Colab mein:
    # !python ai/training/train_exercise_model.py

    clf, prep = train(
        data_path='data/raw/exercise_dataset.csv'
    )