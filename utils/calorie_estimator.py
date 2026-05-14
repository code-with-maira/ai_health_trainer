from utils.constants import MET_VALUES

def estimate_calories(
    duration_min: float,
    weight_kg: float,
    activity: str = "weight_training"
) -> float:
    """
    Calories = MET × weight_kg × duration_hours
    """
    met = MET_VALUES.get(activity, 3.5)
    return round(met * weight_kg * (duration_min / 60), 2)

def calories_from_sets(sets: list[dict], body_weight_kg: float = 70) -> float:
    """Rough estimate from set volume"""
    total_reps = sum(s.get("reps", 0) for s in sets)
    # ~0.15 kcal per rep at avg intensity
    return round(total_reps * 0.15 * (body_weight_kg / 70), 2)