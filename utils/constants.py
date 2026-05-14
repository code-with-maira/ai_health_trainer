# Exercise categories
EXERCISE_CATEGORIES = {
    "push": ["Bench Press", "Shoulder Press", "Push Up", "Tricep Dip"],
    "pull": ["Deadlift", "Pull Up", "Barbell Row", "Lat Pulldown"],
    "legs": ["Squat", "Leg Press", "Lunges", "Romanian Deadlift"],
    "core": ["Plank", "Crunch", "Russian Twist", "Leg Raise"],
}

# MET values for calorie estimation
MET_VALUES = {
    "weight_training": 3.5,
    "cardio_moderate": 7.0,
    "cardio_intense": 10.0,
    "stretching": 2.5,
}

# ACWR thresholds
ACWR_SAFE_MIN = 0.8
ACWR_SAFE_MAX = 1.3
ACWR_HIGH_RISK = 1.5

# Plateau detection
PLATEAU_CV_THRESHOLD = 2.0   # coefficient of variation %
PLATEAU_WINDOW = 5           # last N sessions

# Report
DEFAULT_REPORT_PATH = "reports/"
DEFAULT_CSV_PATH    = "data/sessions.csv"