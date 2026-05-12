# =========================================================
# ai/models/recommendation_model.py
# =========================================================

import numpy as np
import pandas as pd
import pickle
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class RecommendationModel:
    """
    AI Workout Recommendation System
    """

    def __init__(self):

        self.scaler = StandardScaler()

        self.exercise_db = None

        self.feature_matrix = None

        self.is_loaded = False

    # =====================================================
    # LOAD EXERCISE DATABASE
    # =====================================================

    def load_exercises(self, df):

        required_columns = [

            "exercise_name",
            "muscle_group",
            "difficulty",
            "equipment",
            "calories"
        ]

        for col in required_columns:

            if col not in df.columns:

                raise ValueError(
                    f"Missing column: {col}"
                )

        self.exercise_db = df.copy()

        # ---------------------------------------------
        # Encode categorical features
        # ---------------------------------------------

        encoded = pd.get_dummies(

            df[[
                "muscle_group",
                "difficulty",
                "equipment"
            ]]
        )

        # Add calories as numeric feature
        encoded["calories"] = df["calories"]

        self.feature_matrix = self.scaler.fit_transform(
            encoded
        )

        self.is_loaded = True

        print(f"{len(df)} exercises loaded!")

    # =====================================================
    # BUILD USER VECTOR
    # =====================================================

    def build_user_vector(self, profile):

        level_map = {

            "beginner": 0,
            "intermediate": 1,
            "advanced": 2
        }

        fatigue = profile.get(
            "fatigue",
            0.3
        )

        goal = profile.get(
            "goal",
            "weight_loss"
        )

        equipment = profile.get(
            "equipment",
            "none"
        )

        # ---------------------------------------------
        # Goal-based calorie preference
        # ---------------------------------------------

        calorie_pref = {

            "weight_loss": 500,

            "muscle_gain": 350,

            "endurance": 450,

            "mobility": 200

        }.get(goal, 300)

        # ---------------------------------------------
        # Difficulty preference
        # ---------------------------------------------

        level = level_map.get(

            profile.get("level", "beginner"),

            0
        )

        if fatigue > 0.7:
            difficulty = "easy"

        elif level == 0:
            difficulty = "easy"

        elif level == 1:
            difficulty = "medium"

        else:
            difficulty = "hard"

        # ---------------------------------------------
        # Create feature row
        # ---------------------------------------------

        sample = pd.DataFrame([{

            "muscle_group": profile.get(
                "target_muscle",
                "full"
            ),

            "difficulty": difficulty,

            "equipment": equipment,

            "calories": calorie_pref
        }])

        encoded = pd.get_dummies(sample)

        # Align columns with training matrix
        train_cols = pd.get_dummies(
            self.exercise_db[[
                "muscle_group",
                "difficulty",
                "equipment"
            ]]
        ).columns.tolist()

        for col in train_cols:

            if col not in encoded.columns:
                encoded[col] = 0

        encoded = encoded[train_cols]

        encoded["calories"] = calorie_pref

        return self.scaler.transform(encoded)

    # =====================================================
    # RECOMMEND
    # =====================================================

    def recommend(self, user_profile, n=5):

        if not self.is_loaded:
            raise RuntimeError(
                "Exercise database not loaded"
            )

        user_vector = self.build_user_vector(
            user_profile
        )

        similarities = cosine_similarity(

            user_vector,

            self.feature_matrix

        )[0]

        top_idx = np.argsort(
            similarities
        )[-n:][::-1]

        recommendations = []

        for idx in top_idx:

            row = self.exercise_db.iloc[idx]

            recommendations.append({

                "exercise": row["exercise_name"],

                "muscle_group": row["muscle_group"],

                "difficulty": row["difficulty"],

                "equipment": row["equipment"],

                "calories": int(row["calories"]),

                "score": round(
                    float(similarities[idx]),
                    3
                )
            })

        return recommendations

    # =====================================================
    # SAVE
    # =====================================================

    def save(
        self,
        path="saved_models/recommendation_model.pkl"
    ):

        os.makedirs(
            "saved_models",
            exist_ok=True
        )

        with open(path, "wb") as f:

            pickle.dump({

                "exercise_db": self.exercise_db,

                "feature_matrix": self.feature_matrix,

                "scaler": self.scaler

            }, f)

        print(f"Saved -> {path}")

    # =====================================================
    # LOAD
    # =====================================================

    def load(
        self,
        path="saved_models/recommendation_model.pkl"
    ):

        with open(path, "rb") as f:

            data = pickle.load(f)

        self.exercise_db = data["exercise_db"]

        self.feature_matrix = data["feature_matrix"]

        self.scaler = data["scaler"]

        self.is_loaded = True

        print(f"Loaded <- {path}")


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    rec = RecommendationModel()

    # ---------------------------------------------
    # Dummy exercise database
    # ---------------------------------------------

    df = pd.DataFrame({

        "exercise_name": [

            "Squat",
            "Pushup",
            "Plank",
            "Lunge",
            "Burpee",
            "Deadlift"
        ],

        "muscle_group": [

            "legs",
            "chest",
            "core",
            "legs",
            "full",
            "back"
        ],

        "difficulty": [

            "medium",
            "easy",
            "easy",
            "medium",
            "hard",
            "hard"
        ],

        "equipment": [

            "none",
            "none",
            "none",
            "none",
            "none",
            "barbell"
        ],

        "calories": [

            350,
            200,
            120,
            250,
            500,
            450
        ]
    })

    rec.load_exercises(df)

    # ---------------------------------------------
    # User profile
    # ---------------------------------------------

    user = {

        "level": "beginner",

        "goal": "weight_loss",

        "fatigue": 0.2,

        "equipment": "none",

        "target_muscle": "full"
    }

    results = rec.recommend(user)

    print("\nRecommendations:\n")

    for r in results:
        print(r)

    rec.save()