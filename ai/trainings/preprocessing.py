import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os, pickle

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def load_data(self, csv_path):
        """CSV dataset load karo"""
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        return df

    def clean_data(self, df):
        """Missing values aur duplicates hatao"""
        before = len(df)
        df = df.dropna()
        df = df.drop_duplicates()
        after = len(df)
        print(f"Cleaning: {before - after} rows hatayi gayi")
        return df

    def encode_labels(self, y):
        """Text labels ko numbers mein badlo"""
        # e.g. 'squat' -> 0, 'pushup' -> 1
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"Classes: {list(self.label_encoder.classes_)}")
        return y_encoded

    def normalize(self, X_train, X_test=None):
        """Features ko 0-1 range mein lao"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        return X_train_scaled

    def split(self, X, y, test_size=0.2, val_size=0.1):
        """Train, validation, test mein taqseem karo"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )
        print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save(self, path='saved_models/preprocessor.pkl'):
        os.makedirs('saved_models', exist_ok=True)
        pickle.dump({'scaler': self.scaler,
                     'encoder': self.label_encoder}, open(path, 'wb'))
        print(f"Preprocessor saved: {path}")

    def load(self, path='saved_models/preprocessor.pkl'):
        data = pickle.load(open(path, 'rb'))
        self.scaler = data['scaler']
        self.label_encoder = data['encoder']
        print("Preprocessor loaded!")


# Test karo VS Code mein
if __name__ == "__main__":
    prep = Preprocessor()

    # Dummy data se test
    df = pd.DataFrame({
        'feature1': np.random.rand(200),
        'feature2': np.random.rand(200),
        'feature3': np.random.rand(200),
        'label': np.random.choice(['squat','pushup','plank'], 200)
    })

    df = prep.clean_data(df)
    X = df[['feature1','feature2','feature3']].values
    y = prep.encode_labels(df['label'].values)

    X_train, X_val, X_test, y_train, y_val, y_test = prep.split(X, y)
    X_train, X_test = prep.normalize(X_train, X_test)

    print("Preprocessing complete!")
    prep.save()