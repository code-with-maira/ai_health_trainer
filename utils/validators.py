from datetime import datetime

def validate_session(data: dict) -> tuple[bool, str]:
    required = ["exercise", "sets", "total_duration_min"]
    for field in required:
        if field not in data:
            return False, f"Missing field: {field}"

    if not isinstance(data["sets"], list) or len(data["sets"]) == 0:
        return False, "sets must be a non-empty list"

    for i, s in enumerate(data["sets"]):
        if s.get("reps", 0) <= 0:
            return False, f"Set {i+1}: reps must be > 0"
        if s.get("weight", 0) < 0:
            return False, f"Set {i+1}: weight cannot be negative"

    if data["total_duration_min"] <= 0:
        return False, "Duration must be > 0"

    return True, "ok"

def validate_date(date_str: str) -> bool:
    try:
        datetime.fromisoformat(date_str)
        return True
    except ValueError:
        return False

def validate_exercise_name(name: str) -> bool:
    return bool(name and isinstance(name, str) and len(name.strip()) >= 2)