"""
posture_engine.py
-----------------
Production-grade posture analysis engine.

Features:
- Posture classification (6 classes)
- Posture score 0-100
- Named fault detection with severity
- Correction cues
- Confidence scoring

Fixes applied to original PostureEngine:
  FIX 1 — landmarks[11]["x"] dict access → Landmark dataclass (.x .y)
  FIX 2 — Only 1 angle checked (back_angle) → 6 checks (neck, trunk,
           shoulder asymmetry, hip tilt, knee hyperextension)
  FIX 3 — Only 3 string returns → PostureResult dataclass with score,
           faults, corrections, confidence
  FIX 4 — No visibility check → joints checked before use
  FIX 5 — atan2 angle formula unreliable → proper vector dot-product angle
  FIX 6 — No integration with BodySnapshot → takes BodySnapshot directly
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum

from core.landmark_extractor import BodySnapshot


# ================================================================== #
#  Enums
# ================================================================== #

class PostureClass(str, Enum):
    UPRIGHT      = "Upright"
    FORWARD_HEAD = "Forward Head"
    ROUNDED_BACK = "Rounded Back / Kyphosis"
    SWAYBACK     = "Swayback"
    LATERAL_TILT = "Lateral Tilt"
    UNKNOWN      = "Unknown"


# ================================================================== #
#  Data Classes
# ================================================================== #

@dataclass
class PostureFault:
    name:        str
    severity:    str          # "low" | "medium" | "high"
    description: str
    angle_key:   str   = ""
    angle_value: float = -1.0


@dataclass
class PostureResult:
    classification: PostureClass        = PostureClass.UNKNOWN
    score:          float               = 0.0       # 0-100 (100 = perfect)
    faults:         List[PostureFault]  = field(default_factory=list)
    corrections:    List[str]           = field(default_factory=list)
    angles_used:    Dict[str, float]    = field(default_factory=dict)
    confidence:     float               = 0.0       # 0-1

    # Backward-compatible string property
    @property
    def label(self) -> str:
        """Returns simple string label — compatible with old code."""
        if self.score >= 80:
            return "Good Posture"
        elif self.score >= 55:
            return "Slight Lean"
        return "Bad Posture"


# ================================================================== #
#  Thresholds (all tunable)
# ================================================================== #

class PostureThresholds:
    NECK_TILT_WARN    = 15.0   # degrees forward
    NECK_TILT_BAD     = 30.0
    TRUNK_LEAN_WARN   = 160.0  # trunk angle (less = more forward lean)
    TRUNK_LEAN_BAD    = 140.0
    SHOULDER_ASYM_PX  = 10.0   # pixel height difference
    HIP_ASYM_PX       = 8.0
    KNEE_HYPEREXT_DEG = 185.0  # over-extension
    BACK_GOOD         = 160.0  # backward compat with original
    BACK_SLIGHT       = 130.0


# ================================================================== #
#  PostureEngine
# ================================================================== #

class PostureEngine:
    """
    Analyses a BodySnapshot and returns a PostureResult.

    Usage:
        engine = PostureEngine()
        result = engine.analyse(snap)

        # New API
        print(result.score)           # 85.0
        print(result.classification)  # PostureClass.UPRIGHT
        print(result.faults)          # [PostureFault(...)]

        # Old API still works
        print(result.label)           # "Good Posture"
    """

    def __init__(self, thresholds: Optional[PostureThresholds] = None):
        self.th = thresholds or PostureThresholds()

    # ------------------------------------------------------------------ #
    #  Main API — takes BodySnapshot (system-compatible)                  #
    # ------------------------------------------------------------------ #

    def analyse(self, snap: BodySnapshot) -> PostureResult:
        """
        Full posture analysis from BodySnapshot.
        Returns PostureResult with score, classification, faults.
        """
        if snap is None or not snap.fully_visible:
            return PostureResult(confidence=0.0)

        faults:      List[PostureFault] = []
        corrections: List[str]          = []
        angles = snap.angles

        # ---- Neck / Forward Head ------------------------------------
        neck = angles.get("neck_tilt", -1.0)
        if neck >= 0:
            if neck >= self.th.NECK_TILT_BAD:
                faults.append(PostureFault(
                    name="Severe Forward Head",
                    severity="high",
                    description=f"Neck tilt {neck:.1f}° — head far forward.",
                    angle_key="neck_tilt", angle_value=neck,
                ))
                corrections.append(
                    "Tuck chin and draw head back over shoulders."
                )
            elif neck >= self.th.NECK_TILT_WARN:
                faults.append(PostureFault(
                    name="Mild Forward Head",
                    severity="low",
                    description=f"Neck tilt {neck:.1f}° — slight forward lean.",
                    angle_key="neck_tilt", angle_value=neck,
                ))

        # ---- Trunk / Back lean --------------------------------------
        trunk_l = angles.get("trunk_lean_left",  -1.0)
        trunk_r = angles.get("trunk_lean_right", -1.0)
        trunk_avg = self._avg_valid(trunk_l, trunk_r)

        if trunk_avg >= 0:
            if trunk_avg <= self.th.TRUNK_LEAN_BAD:
                faults.append(PostureFault(
                    name="Significant Forward Lean",
                    severity="high",
                    description=f"Trunk angle {trunk_avg:.1f}° — rounded back.",
                    angle_key="trunk_lean_avg", angle_value=trunk_avg,
                ))
                corrections.append(
                    "Extend spine, lift chest, pull shoulder blades together."
                )
            elif trunk_avg <= self.th.TRUNK_LEAN_WARN:
                faults.append(PostureFault(
                    name="Mild Trunk Lean",
                    severity="medium",
                    description=f"Trunk angle {trunk_avg:.1f}° — slight lean.",
                    angle_key="trunk_lean_avg", angle_value=trunk_avg,
                ))
                corrections.append("Engage core and lengthen spine.")

        # FIX 4 — check joint visibility before using
        # ---- Shoulder Asymmetry -------------------------------------
        ls = snap.left_shoulder
        rs = snap.right_shoulder
        if ls and rs:
            asym = abs(ls.py - rs.py)
            if asym > self.th.SHOULDER_ASYM_PX:
                severity = "high" if asym > self.th.SHOULDER_ASYM_PX * 2 else "medium"
                faults.append(PostureFault(
                    name="Shoulder Asymmetry",
                    severity=severity,
                    description=f"Shoulder height diff: {asym:.0f}px — lateral tilt.",
                ))
                corrections.append("Level your shoulders; avoid hiking one side.")

        # ---- Hip Asymmetry ------------------------------------------
        lh = snap.left_hip
        rh = snap.right_hip
        if lh and rh:
            hasym = abs(lh.py - rh.py)
            if hasym > self.th.HIP_ASYM_PX:
                faults.append(PostureFault(
                    name="Hip Tilt",
                    severity="medium",
                    description=f"Hip height diff: {hasym:.0f}px.",
                ))
                corrections.append("Distribute weight evenly on both feet.")

        # ---- Knee Hyperextension ------------------------------------
        for side, key in [("Left", "left_knee"), ("Right", "right_knee")]:
            kangle = angles.get(key, -1.0)
            if kangle > self.th.KNEE_HYPEREXT_DEG:
                faults.append(PostureFault(
                    name=f"{side} Knee Hyperextension",
                    severity="high",
                    description=f"{side} knee {kangle:.1f}° — locking out.",
                    angle_key=key, angle_value=kangle,
                ))
                corrections.append(
                    f"Soften {side.lower()} knee — never lock the joint."
                )

        # ---- Classification -----------------------------------------
        classification = self._classify(faults, neck, trunk_avg)

        # ---- Score (penalty-based) ----------------------------------
        penalty = sum(
            {"low": 5, "medium": 15, "high": 30}.get(f.severity, 0)
            for f in faults
        )
        score = max(0.0, 100.0 - float(penalty))

        # ---- Confidence ---------------------------------------------
        confidence = 1.0 if snap.fully_visible else 0.5

        return PostureResult(
            classification=classification,
            score=round(score, 1),
            faults=faults,
            corrections=corrections,
            angles_used={
                k: round(v, 2)
                for k, v in angles.items()
                if v >= 0
            },
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  Backward-compatible API — takes raw landmark list                  #
    # ------------------------------------------------------------------ #

    def analyze_posture(self, landmarks) -> str:
        """
        Backward-compatible method.
        Accepts old-style landmark list (dict or Landmark dataclass).
        Returns simple string: "Good Posture" | "Slight Lean" | "Bad Posture"

        FIX 1 — supports both dict and dataclass landmark format
        FIX 5 — uses proper vector angle, not atan2 subtraction
        """
        if not landmarks or len(landmarks) < 26:
            return "Unknown"

        try:
            # FIX 1 — support both dict {"x":...} and Landmark dataclass
            def _xy(lm):
                if hasattr(lm, "x"):
                    return (lm.x, lm.y)         # Landmark dataclass
                return (lm["x"], lm["y"])        # dict format

            # FIX 4 — visibility check
            def _vis(lm):
                if hasattr(lm, "visibility"):
                    return lm.visibility
                return lm.get("visibility", 1.0)

            lm11 = landmarks[11]
            lm23 = landmarks[23]
            lm25 = landmarks[25]

            if _vis(lm11) < 0.3 or _vis(lm23) < 0.3 or _vis(lm25) < 0.3:
                return "Unknown"

            shoulder = _xy(lm11)
            hip      = _xy(lm23)
            knee     = _xy(lm25)

            # FIX 5 — proper vector dot-product angle (not atan2 subtraction)
            back_angle = self.calculate_angle(shoulder, hip, knee)

            if back_angle > self.th.BACK_GOOD:
                return "Good Posture"
            elif back_angle > self.th.BACK_SLIGHT:
                return "Slight Lean"
            return "Bad Posture"

        except (IndexError, KeyError, AttributeError):
            return "Unknown"

    # ------------------------------------------------------------------ #
    #  Angle Calculation                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_angle(
        a: tuple,
        b: tuple,
        c: tuple,
    ) -> float:
        """
        Angle at point B in the A-B-C triplet (degrees).

        FIX 5 — uses vector dot-product (accurate, no wrap-around issues)
        vs original atan2 subtraction which could give wrong results
        near 0°/360° boundary.

        Args:
            a, b, c : (x, y) tuples
        """
        ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
        bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)

        norm = np.linalg.norm(ba) * np.linalg.norm(bc)
        if norm < 1e-6:
            return 0.0

        cos_a = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_a)))

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _classify(
        self,
        faults:    List[PostureFault],
        neck_tilt: float,
        trunk_avg: float,
    ) -> PostureClass:

        names = {f.name for f in faults}

        if "Shoulder Asymmetry" in names or "Hip Tilt" in names:
            return PostureClass.LATERAL_TILT

        if "Significant Forward Lean" in names or "Severe Forward Head" in names:
            if neck_tilt >= self.th.NECK_TILT_BAD:
                return PostureClass.FORWARD_HEAD
            return PostureClass.ROUNDED_BACK

        if "Mild Forward Head" in names:
            return PostureClass.FORWARD_HEAD

        if not faults:
            return PostureClass.UPRIGHT

        return PostureClass.UNKNOWN

    @staticmethod
    def _avg_valid(*values: float) -> float:
        valid = [v for v in values if v >= 0]
        return float(np.mean(valid)) if valid else -1.0