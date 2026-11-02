"""
core/alert_manager.py
──────────────────────
Non-spammy alert system.

Alerts are event-based — once a fire/hazard event begins, a single alert is
emitted.  While the event is active, no duplicate alerts are sent.  A new
alert fires only when the hazard escalates or clears.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from core.hazard_engine import HazardLevel, HazardSnapshot


@dataclass
class Alert:
    """A single alert notification."""
    timestamp:   str
    level:       str
    title:       str
    message:     str
    hazard_level: str

    def to_dict(self) -> dict:
        return {
            "timestamp":   self.timestamp,
            "level":       self.level,
            "title":       self.title,
            "message":     self.message,
            "hazard_level": self.hazard_level,
        }


class AlertManager:
    """
    Produces alerts based on hazard state transitions.

    Rules
    -----
    - One alert per event start.
    - One alert per escalation (hazard level goes up).
    - One alert when event clears.
    - No per-frame spam.
    """

    def __init__(self) -> None:
        self._last_hazard: Optional[HazardLevel] = None
        self._alert_history: list[Alert] = []

    def process(self, snapshot: HazardSnapshot) -> Optional[Alert]:
        """
        Evaluate the snapshot and emit an alert if warranted.

        Returns
        -------
        Alert or None
        """
        current = snapshot.hazard_level
        prev    = self._last_hazard

        alert = None

        if prev is None or prev == HazardLevel.NORMAL:
            if current != HazardLevel.NORMAL:
                alert = self._make_alert(snapshot, "Event started")
        elif current.numeric > prev.numeric:
            alert = self._make_alert(snapshot, "Hazard escalated")
        elif current == HazardLevel.NORMAL and prev != HazardLevel.NORMAL:
            alert = Alert(
                timestamp=self._now(),
                level="info",
                title="Situation Cleared",
                message="Hazard status returned to NORMAL.",
                hazard_level=current.value,
            )

        self._last_hazard = current

        if alert:
            self._alert_history.append(alert)

        return alert

    def get_history(self) -> list[dict]:
        return [a.to_dict() for a in reversed(self._alert_history)]

    def reset(self) -> None:
        self._last_hazard = None

    # ── Private ───────────────────────────────────────────────────────────────

    def _make_alert(self, snapshot: HazardSnapshot, reason: str) -> Alert:
        parts = []
        if snapshot.fire_state.value in ("Confirmed", "Emergency"):
            parts.append("🚨 Fire detected")
        if snapshot.smoke_analysis and snapshot.smoke_analysis.detected:
            sa = snapshot.smoke_analysis
            parts.append(
                f"Smoke: {sa.hazard_level.value} | "
                f"Density: {sa.density_index:.0f}/100 | "
                f"Coverage: {sa.coverage_pct:.0f}%"
            )
        if snapshot.people_count > 0:
            parts.append(f"People: {snapshot.people_count}")
        if snapshot.livestock_count > 0:
            parts.append(f"Livestock: {snapshot.livestock_count}")

        level_labels = {
            HazardLevel.LOW:      "info",
            HazardLevel.MODERATE: "warning",
            HazardLevel.HIGH:     "danger",
            HazardLevel.CRITICAL: "critical",
        }

        return Alert(
            timestamp=self._now(),
            level=level_labels.get(snapshot.hazard_level, "info"),
            title=f"{snapshot.hazard_level.value} HAZARD — {reason}",
            message=" | ".join(parts) if parts else reason,
            hazard_level=snapshot.hazard_level.value,
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
