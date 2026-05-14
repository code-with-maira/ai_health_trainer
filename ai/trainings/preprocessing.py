import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os, pickle

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    # ─────────────────────────────
    # LOAD DATA
    # ─────────────────────────────
    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        return df

    # ─────────────────────────────
    # CLEAN DATA
    # ─────────────────────────────
    def clean_data(self, df):
        before = len(df)
        df = df.dropna()
        df = df.drop_duplicates()
        after = len(df)
        print(f"Cleaning: {before - after} rows hatayi gayi")
        return df

    # ─────────────────────────────
    # LABEL ENCODING
    # ─────────────────────────────
    def encode_labels(self, y):
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"Classes: {list(self.label_encoder.classes_)}")
        return y_encoded

    # ─────────────────────────────
    # SPLIT DATA
    # ─────────────────────────────
    def split(self, X, y, test_size=0.2, val_size=0.1):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )
        print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ─────────────────────────────
    # NORMALIZATION (FIXED)
    # ─────────────────────────────
    def normalize(self, X_train, X_test=None):
        # 🔥 FORCE numeric conversion (IMPORTANT FIX)
        X_train = np.array(X_train, dtype=np.float32)

        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_test is not None:
            X_test = np.array(X_test, dtype=np.float32)
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled

        return X_train_scaled

    # ─────────────────────────────
    # SAVE
    # ─────────────────────────────
    def save(self, path='saved_models/preprocessor.pkl'):
        os.makedirs('saved_models', exist_ok=True)
        pickle.dump({
            'scaler': self.scaler,
            'encoder': self.label_encoder
        }, open(path, 'wb'))
        print(f"Preprocessor saved: {path}")

    # ─────────────────────────────
    # LOAD
    # ─────────────────────────────
    def load(self, path='saved_models/preprocessor.pkl'):
        data = pickle.load(open(path, 'rb'))
        self.scaler = data['scaler']
        self.label_encoder = data['encoder']
        print("Preprocessor loaded!")
