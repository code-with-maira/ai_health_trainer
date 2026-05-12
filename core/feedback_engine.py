"""
feedback_engine.py
------------------
Advanced AI-powered Feedback Engine using Claude API.

Features:
- Real-time AI coaching via Claude API (non-blocking)
- Rule-based instant feedback (no API delay)
- Posture, motion, fatigue, injury all considered
- Priority + severity system
- Exercise-specific feedback
- Personalized coaching messages
- Feedback history tracking
"""

import threading
import time
import logging
import requests

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum

from core.posture_engine import PostureResult, PostureClass
from core.motion_analysis import MotionState, MovementPattern, MovementPhase
from core.landmark_extractor import BodySnapshot

logger = logging.getLogger(__name__)


# ===================================================================== #
# ENUMS
# ===================================================================== #

class FeedbackSeverity(str, Enum):
    INFO    = "info"
    WARNING = "warning"
    ALERT   = "alert"


class FeedbackCategory(str, Enum):
    FORM     = "form"
    DEPTH    = "depth"
    SYMMETRY = "symmetry"
    POSTURE  = "posture"
    FATIGUE  = "fatigue"
    INJURY   = "injury"
    GENERAL  = "general"
    AI       = "ai"


# ===================================================================== #
# FEEDBACK ITEM
# ===================================================================== #

@dataclass
class FeedbackItem:

    message:  str
    category: FeedbackCategory  = FeedbackCategory.GENERAL
    severity: FeedbackSeverity  = FeedbackSeverity.INFO
    source:   str               = "rule"  # "rule" or "ai"


# ===================================================================== #
# FEEDBACK BUNDLE
# ===================================================================== #

@dataclass
class FeedbackBundle:

    items:      List[FeedbackItem] = field(default_factory=list)
    ai_message: str                = ""
    timestamp:  float              = 0.0

    @property
    def primary(self) -> Optional[FeedbackItem]:
        """Highest severity item."""
        if not self.items:
            return None
        order = {
            FeedbackSeverity.ALERT:   0,
            FeedbackSeverity.WARNING: 1,
            FeedbackSeverity.INFO:    2,
        }
        return min(self.items, key=lambda x: order[x.severity])

    @property
    def alerts(self) -> List[FeedbackItem]:
        return [i for i in self.items if i.severity == FeedbackSeverity.ALERT]

    @property
    def warnings(self) -> List[FeedbackItem]:
        return [i for i in self.items if i.severity == FeedbackSeverity.WARNING]


# ===================================================================== #
# RULE-BASED FEEDBACK
# ===================================================================== #

class RuleEngine:
    """
    Instant rule-based feedback — no API delay.
    Covers form, depth, symmetry, posture.
    """

    def evaluate(
        self,
        snap:    Optional[BodySnapshot],
        motion:  Optional[MotionState],
        posture: Optional[PostureResult],
        injuries: List = [],
    ) -> List[FeedbackItem]:

        items = []

        if snap is None or motion is None:
            return items

        angles  = snap.angles
        pattern = motion.pattern

        # ------------------------------------------------------------ #
        # Injury alerts — highest priority
        # ------------------------------------------------------------ #

        for injury in injuries:
            items.append(FeedbackItem(
                message  = getattr(injury, "message", str(injury)),
                category = FeedbackCategory.INJURY,
                severity = FeedbackSeverity.ALERT,
                source   = "rule",
            ))

        # ------------------------------------------------------------ #
        # Posture
        # ------------------------------------------------------------ #

        if posture:

            if posture.classification == PostureClass.FORWARD_HEAD:
                items.append(FeedbackItem(
                    message  = "Chin up — head is too far forward",
                    category = FeedbackCategory.POSTURE,
                    severity = FeedbackSeverity.WARNING,
                ))

            elif posture.classification == PostureClass.ROUNDED_BACK:
                items.append(FeedbackItem(
                    message  = "Straighten your back — avoid rounding",
                    category = FeedbackCategory.POSTURE,
                    severity = FeedbackSeverity.WARNING,
                ))

            elif posture.classification == PostureClass.LATERAL_TILT:
                items.append(FeedbackItem(
                    message  = "Keep your shoulders level",
                    category = FeedbackCategory.POSTURE,
                    severity = FeedbackSeverity.WARNING,
                ))

            elif posture.classification == PostureClass.SWAYBACK:
                items.append(FeedbackItem(
                    message  = "Engage your core — avoid swayback",
                    category = FeedbackCategory.POSTURE,
                    severity = FeedbackSeverity.ALERT,
                ))

        # ------------------------------------------------------------ #
        # Symmetry
        # ------------------------------------------------------------ #

        if motion.symmetry_score is not None and motion.symmetry_score > 15:
            items.append(FeedbackItem(
                message  = f"Uneven movement — {motion.symmetry_score:.0f}° knee difference",
                category = FeedbackCategory.SYMMETRY,
                severity = FeedbackSeverity.WARNING,
            ))

        # ------------------------------------------------------------ #
        # Exercise-specific form checks
        # ------------------------------------------------------------ #

        if pattern == MovementPattern.SQUAT:
            items += self._squat_feedback(angles, motion)

        elif pattern == MovementPattern.PUSH_UP:
            items += self._pushup_feedback(angles, motion)

        elif pattern == MovementPattern.CURL:
            items += self._curl_feedback(angles, motion)

        elif pattern == MovementPattern.DEADLIFT:
            items += self._deadlift_feedback(angles, snap)

        elif pattern == MovementPattern.LUNGE:
            items += self._lunge_feedback(angles)

        elif pattern == MovementPattern.PRESS:
            items += self._press_feedback(angles)

        # ------------------------------------------------------------ #
        # General good form
        # ------------------------------------------------------------ #

        if not items and motion.is_moving:
            items.append(FeedbackItem(
                message  = "Good form — keep it up!",
                category = FeedbackCategory.GENERAL,
                severity = FeedbackSeverity.INFO,
            ))

        return items

    # ----------------------------------------------------------------- #

    def _squat_feedback(
        self,
        angles: Dict[str, float],
        motion: MotionState,
    ) -> List[FeedbackItem]:

        items = []

        lk = angles.get("left_knee",  -1)
        rk = angles.get("right_knee", -1)
        lh = angles.get("left_hip",   -1)

        knee_avg = (lk + rk) / 2 if lk > 0 and rk > 0 else -1

        if motion.phase == MovementPhase.CONCENTRIC:

            if knee_avg > 0 and knee_avg > 100:
                items.append(FeedbackItem(
                    message  = "Go deeper — aim for 90° at the knee",
                    category = FeedbackCategory.DEPTH,
                    severity = FeedbackSeverity.INFO,
                ))

        if lh > 0 and lh < 70:
            items.append(FeedbackItem(
                message  = "Keep chest up — avoid excessive forward lean",
                category = FeedbackCategory.FORM,
                severity = FeedbackSeverity.WARNING,
            ))

        trunk = angles.get("trunk_lean_avg", -1)

        if trunk > 0 and trunk < 60:
            items.append(FeedbackItem(
                message  = "Back is leaning too far forward",
                category = FeedbackCategory.FORM,
                severity = FeedbackSeverity.WARNING,
            ))

        return items

    # ----------------------------------------------------------------- #

    def _pushup_feedback(
        self,
        angles: Dict[str, float],
        motion: MotionState,
    ) -> List[FeedbackItem]:

        items = []

        le = angles.get("left_elbow",  -1)
        re = angles.get("right_elbow", -1)

        elbow_avg = (le + re) / 2 if le > 0 and re > 0 else -1

        if motion.phase == MovementPhase.CONCENTRIC:

            if elbow_avg > 0 and elbow_avg > 90:
                items.append(FeedbackItem(
                    message  = "Go lower — chest closer to the ground",
                    category = FeedbackCategory.DEPTH,
                    severity = FeedbackSeverity.INFO,
                ))

            elif elbow_avg > 0 and elbow_avg < 70:
                items.append(FeedbackItem(
                    message  = "Great depth!",
                    category = FeedbackCategory.DEPTH,
                    severity = FeedbackSeverity.INFO,
                ))

        trunk = angles.get("trunk_lean_avg", -1)

        if trunk > 0 and (trunk < 75 or trunk > 115):
            items.append(FeedbackItem(
                message  = "Keep body straight — engage your core",
                category = FeedbackCategory.FORM,
                severity = FeedbackSeverity.WARNING,
            ))

        return items

    # ----------------------------------------------------------------- #

    def _curl_feedback(
        self,
        angles: Dict[str, float],
        motion: MotionState,
    ) -> List[FeedbackItem]:

        items = []

        le = angles.get("left_elbow",  -1)
        re = angles.get("right_elbow", -1)

        elbow_avg = (le + re) / 2 if le > 0 and re > 0 else -1

        ls = angles.get("left_shoulder",  -1)
        rs = angles.get("right_shoulder", -1)

        shoulder_avg = (ls + rs) / 2 if ls > 0 and rs > 0 else -1

        if motion.phase == MovementPhase.CONCENTRIC:

            if elbow_avg > 0 and elbow_avg > 100:
                items.append(FeedbackItem(
                    message  = "Curl higher — squeeze at the top",
                    category = FeedbackCategory.DEPTH,
                    severity = FeedbackSeverity.INFO,
                ))

        if shoulder_avg > 0 and shoulder_avg > 30:
            items.append(FeedbackItem(
                message  = "Keep elbows at your sides — avoid swinging",
                category = FeedbackCategory.FORM,
                severity = FeedbackSeverity.WARNING,
            ))

        return items

    # ----------------------------------------------------------------- #

    def _deadlift_feedback(
        self,
        angles: Dict[str, float],
        snap:   BodySnapshot,
    ) -> List[FeedbackItem]:

        items = []

        lh = angles.get("left_hip", -1)

        if lh > 0 and lh < 60:
            items.append(FeedbackItem(
                message  = "Keep your back flat — avoid rounding the lower back",
                category = FeedbackCategory.FORM,
                severity = FeedbackSeverity.ALERT,
            ))

        trunk = angles.get("trunk_lean_avg", -1)

        if trunk > 0 and trunk < 50:
            items.append(FeedbackItem(
                message  = "Hinge at the hips — keep chest up",
                category = FeedbackCategory.FORM,
                severity = FeedbackSeverity.WARNING,
            ))

        return items

    # ----------------------------------------------------------------- #

    def _lunge_feedback(
        self,
        angles: Dict[str, float],
    ) -> List[FeedbackItem]:

        items = []

        lk = angles.get("left_knee",  -1)
        rk = angles.get("right_knee", -1)

        for side, angle in [("front", lk), ("back", rk)]:

            if angle > 0 and angle > 110:
                items.append(FeedbackItem(
                    message  = f"Bend {side} knee more — aim for 90°",
                    category = FeedbackCategory.DEPTH,
                    severity = FeedbackSeverity.INFO,
                ))

        return items

    # ----------------------------------------------------------------- #

    def _press_feedback(
        self,
        angles: Dict[str, float],
    ) -> List[FeedbackItem]:

        items = []

        ls = angles.get("left_shoulder",  -1)
        rs = angles.get("right_shoulder", -1)

        shoulder_avg = (ls + rs) / 2 if ls > 0 and rs > 0 else -1

        if shoulder_avg > 0 and shoulder_avg < 150:
            items.append(FeedbackItem(
                message  = "Press all the way up — full extension",
                category = FeedbackCategory.DEPTH,
                severity = FeedbackSeverity.INFO,
            ))

        return items


# ===================================================================== #
# AI FEEDBACK ENGINE (Claude API)
# ===================================================================== #

class AIFeedbackEngine:
    """
    Non-blocking Claude API coaching — runs in background thread.
    """

    CLAUDE_URL = "https://api.anthropic.com/v1/messages"

    SYSTEM_PROMPT = (
        "You are an expert AI fitness coach. "
        "Analyze the exercise data and give 1-2 sentences of specific, "
        "actionable coaching. Be encouraging but direct. "
        "Focus on the most important correction or motivation. "
        "No lists, no headers — plain coaching text only."
    )

    def __init__(self, api_key: str, interval: float = 5.0):

        self._api_key    = api_key
        self._interval   = interval

        self._message:   str   = ""
        self._lock             = threading.Lock()
        self._last_call: float = 0.0
        self._calling:   bool  = False

    # ----------------------------------------------------------------- #

    def request(
        self,
        snap:     Optional[BodySnapshot],
        motion:   Optional[MotionState],
        posture:  Optional[PostureResult],
        injuries: List = [],
    ):
        """Non-blocking — triggers API call in background if interval passed."""

        if not self._api_key or snap is None:
            return

        now = time.perf_counter()

        if self._calling or (now - self._last_call) < self._interval:
            return

        self._calling    = True
        self._last_call  = now

        prompt = self._build_prompt(snap, motion, posture, injuries)

        thread = threading.Thread(
            target = self._call_api,
            args   = (prompt,),
            daemon = True,
        )
        thread.start()

    # ----------------------------------------------------------------- #

    def get_message(self) -> str:
        with self._lock:
            return self._message

    # ----------------------------------------------------------------- #

    def _build_prompt(
        self,
        snap:     BodySnapshot,
        motion:   Optional[MotionState],
        posture:  Optional[PostureResult],
        injuries: List,
    ) -> str:

        parts = ["Real-time exercise data:"]

        if motion:
            parts.append(
                f"Exercise: {motion.pattern.value}, "
                f"Phase: {motion.phase.value}, "
                f"Reps completed: {motion.rep_count}"
            )

        key_angles = {
            k: v for k, v in snap.angles.items()
            if v > 0 and k in (
                "left_knee", "right_knee",
                "left_elbow", "right_elbow",
                "left_hip", "right_hip",
                "trunk_lean_avg", "neck_tilt",
            )
        }

        if key_angles:
            parts.append(
                "Joint angles: " + ", ".join(
                    f"{k.replace('_', ' ')}: {v:.0f}°"
                    for k, v in key_angles.items()
                )
            )

        if posture:
            parts.append(
                f"Posture: {posture.classification.value}, "
                f"score: {posture.score:.0f}/100"
            )

        if motion and motion.symmetry_score is not None:
            parts.append(
                f"Knee symmetry difference: {motion.symmetry_score:.1f}°"
            )

        if injuries:
            parts.append(
                "Injury alerts: " + ", ".join(
                    getattr(i, "message", str(i)) for i in injuries
                )
            )

        parts.append("Give 1-2 sentences of coaching feedback.")

        return "\n".join(parts)

    # ----------------------------------------------------------------- #

    def _call_api(self, prompt: str):

        try:

            response = requests.post(
                self.CLAUDE_URL,
                headers = {
                    "Content-Type":      "application/json",
                    "x-api-key":         self._api_key,
                    "anthropic-version": "2023-06-01",
                },
                json = {
                    "model":      "claude-sonnet-4-20250514",
                    "max_tokens": 120,
                    "system":     self.SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                },
                timeout = 8,
            )

            if response.status_code == 200:

                text = (
                    response.json()
                    .get("content", [{}])[0]
                    .get("text", "")
                    .strip()
                )

                if text:
                    with self._lock:
                        self._message = text

            else:
                logger.warning(
                    "Claude API %d: %s",
                    response.status_code,
                    response.text[:200],
                )

        except Exception as e:
            logger.error("Claude API error: %s", e)

        finally:
            self._calling = False


# ===================================================================== #
# FEEDBACK ENGINE (main)
# ===================================================================== #

class FeedbackEngine:
    """
    Combined rule-based + AI feedback engine.

    Rule engine gives instant feedback every frame.
    AI engine gives deeper coaching every few seconds.
    """

    def __init__(
        self,
        anthropic_api_key: str   = "",
        ai_interval:       float = 5.0,
    ):

        self._rules = RuleEngine()

        self._ai = AIFeedbackEngine(
            api_key  = anthropic_api_key,
            interval = ai_interval,
        )

        self._history: List[FeedbackBundle] = []

    # ================================================================= #

    def generate(
        self,
        posture:  Optional[PostureResult] = None,
        motion:   Optional[MotionState]   = None,
        fatigue                           = None,
        injuries: List                    = [],
        snap:     Optional[BodySnapshot]  = None,
    ) -> FeedbackBundle:

        # Rule-based — instant
        items = self._rules.evaluate(
            snap     = snap,
            motion   = motion,
            posture  = posture,
            injuries = injuries,
        )

        # Fatigue feedback
        if fatigue is not None:

            fatigue_level = getattr(fatigue, "level", None)

            if fatigue_level and str(fatigue_level).lower() in ("high", "severe"):
                items.append(FeedbackItem(
                    message  = "You look tired — consider resting or slowing down",
                    category = FeedbackCategory.FATIGUE,
                    severity = FeedbackSeverity.WARNING,
                ))

        # AI — non-blocking background call
        self._ai.request(
            snap     = snap,
            motion   = motion,
            posture  = posture,
            injuries = injuries,
        )

        bundle = FeedbackBundle(
            items      = items,
            ai_message = self._ai.get_message(),
            timestamp  = time.time(),
        )

        # Keep last 30 bundles for history/analytics
        self._history.append(bundle)

        if len(self._history) > 30:
            self._history.pop(0)

        return bundle

    # ================================================================= #

    @property
    def history(self) -> List[FeedbackBundle]:
        return self._history

    def reset(self):
        self._history.clear()