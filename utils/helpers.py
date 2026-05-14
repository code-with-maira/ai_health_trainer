import uuid
from datetime import datetime

def generate_id() -> str:
    return str(uuid.uuid4())[:8]

def now_iso() -> str:
    return datetime.now().isoformat()

def format_duration(minutes: float) -> str:
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h}h {m}m" if h > 0 else f"{m}m"

def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def safe_int(val, default=0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default

def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))