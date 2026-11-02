"""
core/hazard_engine.py
──────────────────────
Temporal Hazard Engine.

Aggregates detection signals over time and produces an overall Hazard Status.
Requires temporal consistency before escalating — a single ambiguous frame does
not trigger an emergency state.

Hazard states: NORMAL → LOW → MODERATE → HIGH → CRITICAL

⚠️  This is a situational-awareness indicator only.
     Never interpret NORMAL as "safe to enter."
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from core.detector import Detection, DetectionQuery
from core.smoke_analyzer import SmokeAnalysis, SmokeHazardLevel


# ──────────────────────────────────────────────────────────────────────────────
# Hazard levels
# ──────────────────────────────────────────────────────────────────────────────

class HazardLevel(str, Enum):
    NORMAL   = "NORMAL"
    LOW      = "LOW"
    MODERATE = "MODERATE"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def numeric(self) -> int:
        return {
            "NORMAL":   0,
            "LOW":      1,
            "MODERATE": 2,
            "HIGH":     3,
            "CRITICAL": 4,
        }[self.value]

    @property
    def color_key(self) -> str:
        return {
            "NORMAL":   "color-safe",
            "LOW":      "color-warn-low",
            "MODERATE": "color-warn-moderate",
            "HIGH":     "color-warn-high",
            "CRITICAL": "color-critical",
        }[self.value]


class DetectionState(str, Enum):
    """Fine-grained temporal state for a single detection category."""
    NORMAL            = "Normal"
    SUSPECTED         = "Suspected"
    CONFIRMED         = "Confirmed"
    EMERGENCY         = "Emergency"


# ──────────────────────────────────────────────────────────────────────────────
# Result snapshot
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HazardSnapshot:
    """Complete hazard assessment for one moment in time."""
    hazard_level: HazardLevel
    fire_state: DetectionState
    smoke_state: DetectionState
    people_count: int
    livestock_count: int
    smoke_analysis: Optional[SmokeAnalysis]
    frame_index: int
    alerts: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"HAZARD  {self.hazard_level.value}",
            f"FIRE    {self.fire_state.value}",
            f"SMOKE   {self.smoke_state.value}",
            f"PEOPLE  {self.people_count}",
            f"LIVESTOCK {self.livestock_count}",
        ]
        if self.smoke_analysis and self.smoke_analysis.detected:
            lines.append(str(self.smoke_analysis))
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Persistence tracker
# ──────────────────────────────────────────────────────────────────────────────

class _PersistenceTracker:
    """Tracks detection presence/absence over a sliding window."""

    def __init__(
        self,
        window: int = 10,
        confirm_threshold: float = 0.6,
        clear_threshold: float = 0.2,
    ) -> None:
        self._window = window
        self._confirm_threshold = confirm_threshold
        self._clear_threshold = clear_threshold
        self._history: deque[bool] = deque(maxlen=window)
        self._state = DetectionState.NORMAL

    def update(self, detected: bool) -> DetectionState:
        self._history.append(detected)
        if len(self._history) < 2:
            return self._state

        rate = sum(self._history) / len(self._history)

        if rate >= self._confirm_threshold:
            if self._state == DetectionState.NORMAL:
                self._state = DetectionState.SUSPECTED
            elif self._state == DetectionState.SUSPECTED:
                self._state = DetectionState.CONFIRMED
            elif self._state == DetectionState.CONFIRMED:
                self._state = DetectionState.EMERGENCY
        elif rate <= self._clear_threshold:
            # Gradual de-escalation
            if self._state == DetectionState.EMERGENCY:
                self._state = DetectionState.CONFIRMED
            elif self._state == DetectionState.CONFIRMED:
                self._state = DetectionState.SUSPECTED
            elif self._state == DetectionState.SUSPECTED:
                self._state = DetectionState.NORMAL

        return self._state

    @property
    def state(self) -> DetectionState:
        return self._state

    def reset(self) -> None:
        self._history.clear()
        self._state = DetectionState.NORMAL


# ──────────────────────────────────────────────────────────────────────────────
# Hazard Engine
# ──────────────────────────────────────────────────────────────────────────────

class HazardEngine:
    """
    Aggregates detection results over time into a single hazard assessment.

    Parameters
    ----------
    window : int
        Sliding window length (frames) for persistence tracking.
    confirm_threshold : float
        Fraction of recent frames with detection required to confirm.
    clear_threshold : float
        Fraction below which a detection is considered cleared.
    """

    def __init__(
        self,
        window: int = 10,
        confirm_threshold: float = 0.6,
        clear_threshold: float = 0.2,
    ) -> None:
        self._fire_tracker  = _PersistenceTracker(window, confirm_threshold, clear_threshold)
        self._smoke_tracker = _PersistenceTracker(window, confirm_threshold, clear_threshold)
        self._frame_index   = 0
        self._last_people_count    = 0
        self._last_livestock_count = 0

    def update(
        self,
        fire_detection:      Detection,
        smoke_detection:     Detection,
        people_detection:    Detection,
        livestock_detection: Detection,
        smoke_analysis:      Optional[SmokeAnalysis] = None,
    ) -> HazardSnapshot:
        """
        Feed one set of detection results and compute the current hazard snapshot.
        """
        self._frame_index += 1

        fire_state  = self._fire_tracker.update(fire_detection.detected)
        smoke_state = self._smoke_tracker.update(smoke_detection.detected)

        self._last_people_count    = people_detection.count
        self._last_livestock_count = livestock_detection.count

        hazard_level = self._compute_hazard(
            fire_state, smoke_state, smoke_analysis,
            people_detection.count, livestock_detection.count,
        )

        alerts = self._build_alerts(fire_state, smoke_state, hazard_level)

        return HazardSnapshot(
            hazard_level=hazard_level,
            fire_state=fire_state,
            smoke_state=smoke_state,
            people_count=people_detection.count,
            livestock_count=livestock_detection.count,
            smoke_analysis=smoke_analysis,
            frame_index=self._frame_index,
            alerts=alerts,
        )

    def reset(self) -> None:
        self._fire_tracker.reset()
        self._smoke_tracker.reset()
        self._frame_index = 0
        self._last_people_count = 0
        self._last_livestock_count = 0

    # ── Private ───────────────────────────────────────────────────────────────

    def _compute_hazard(
        self,
        fire_state:  DetectionState,
        smoke_state: DetectionState,
        smoke_analysis: Optional[SmokeAnalysis],
        people_count: int,
        livestock_count: int,
    ) -> HazardLevel:
        score = 0

        # Fire contribution
        fire_scores = {
            DetectionState.NORMAL:    0,
            DetectionState.SUSPECTED: 2,
            DetectionState.CONFIRMED: 3,
            DetectionState.EMERGENCY: 4,
        }
        score += fire_scores[fire_state]

        # Smoke contribution
        if smoke_analysis and smoke_analysis.detected:
            smoke_level_scores = {
                SmokeHazardLevel.NONE:     0,
                SmokeHazardLevel.VERY_LOW: 1,
                SmokeHazardLevel.LOW:      1,
                SmokeHazardLevel.MODERATE: 2,
                SmokeHazardLevel.HIGH:     3,
                SmokeHazardLevel.CRITICAL: 4,
            }
            score += smoke_level_scores.get(smoke_analysis.hazard_level, 0)
        elif smoke_state in (DetectionState.CONFIRMED, DetectionState.EMERGENCY):
            score += 2

        # People/livestock add urgency but don't directly escalate hazard
        if people_count > 0:
            score += 1
        if livestock_count > 0:
            score += 1

        # Map score to hazard level
        if score <= 0:
            return HazardLevel.NORMAL
        if score <= 2:
            return HazardLevel.LOW
        if score <= 4:
            return HazardLevel.MODERATE
        if score <= 6:
            return HazardLevel.HIGH
        return HazardLevel.CRITICAL

    def _build_alerts(
        self,
        fire_state:  DetectionState,
        smoke_state: DetectionState,
        hazard_level: HazardLevel,
    ) -> list[str]:
        """Generate alert strings — called once per snapshot."""
        alerts = []
        if fire_state == DetectionState.CONFIRMED:
            alerts.append("🚨 FIRE CONFIRMED")
        elif fire_state == DetectionState.EMERGENCY:
            alerts.append("🚨 FIRE — EMERGENCY")
        if smoke_state == DetectionState.CONFIRMED:
            alerts.append("⚠ SMOKE CONFIRMED")
        if hazard_level == HazardLevel.CRITICAL:
            alerts.append("🔴 CRITICAL HAZARD")
        return alerts
