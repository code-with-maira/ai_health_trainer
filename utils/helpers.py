# utils/helpers.py

import uuid
from datetime import datetime


class Helpers:

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())[:8]

    @staticmethod
    def now_iso() -> str:
        return datetime.now().isoformat()

    @staticmethod
    def format_duration(minutes: float) -> str:

        h = int(minutes // 60)
        m = int(minutes % 60)

        return f"{h}h {m}m" if h > 0 else f"{m}m"

    @staticmethod
    def safe_float(val, default=0.0) -> float:

        try:
            return float(val)

        except (TypeError, ValueError):
            return default

    @staticmethod
    def safe_int(val, default=0) -> int:

        try:
            return int(val)

        except (TypeError, ValueError):
            return default

    @staticmethod
    def clamp(value: float,
              min_val: float,
              max_val: float) -> float:

        return max(min_val, min(max_val, value))