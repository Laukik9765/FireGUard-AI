"""
core/event_manager.py
──────────────────────
Manages detection events and their lifecycle.

An "event" is a confirmed hazard episode — e.g., a confirmed fire detection
that persists for multiple frames.  Events are:
  - Created when a new hazard state is confirmed
  - Updated while the hazard persists
  - Closed when the hazard clears

Events are saved to outputs/events/ as JSON for the Events history screen.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.hazard_engine import DetectionState, HazardLevel, HazardSnapshot


EVENTS_DIR = Path("outputs/events")


@dataclass
class FireGuardEvent:
    """A single hazard event with start/end timestamps."""
    event_id:    str
    started_at:  str
    ended_at:    Optional[str]
    peak_hazard: str
    fire_confirmed:  bool
    smoke_confirmed: bool
    max_people:      int
    max_livestock:   int
    peak_density:    float
    frame_count:     int = 0
    closed:          bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class EventManager:
    """
    Creates, updates, and closes FireGuard events.

    Persists events to disk as JSON.
    """

    def __init__(self, events_dir: Path = EVENTS_DIR) -> None:
        self._events_dir = Path(events_dir)
        self._events_dir.mkdir(parents=True, exist_ok=True)
        self._active_event: Optional[FireGuardEvent] = None

    def process(self, snapshot: HazardSnapshot) -> Optional[FireGuardEvent]:
        """
        Feed a hazard snapshot and update event state.

        Returns the current event (open or just-closed) if one exists.
        """
        is_hazard = snapshot.hazard_level != HazardLevel.NORMAL

        if is_hazard and self._active_event is None:
            self._active_event = self._open_event(snapshot)
        elif is_hazard and self._active_event is not None:
            self._update_event(snapshot)
        elif not is_hazard and self._active_event is not None:
            self._close_event()

        return self._active_event

    @property
    def active_event(self) -> Optional[FireGuardEvent]:
        return self._active_event

    def list_events(self) -> list[dict]:
        """Load all saved events from disk (most recent first)."""
        paths = sorted(self._events_dir.glob("*.json"), reverse=True)
        events = []
        for p in paths:
            try:
                events.append(json.loads(p.read_text()))
            except Exception:
                pass
        return events

    # ── Private ───────────────────────────────────────────────────────────────

    def _open_event(self, snapshot: HazardSnapshot) -> FireGuardEvent:
        event = FireGuardEvent(
            event_id=str(uuid.uuid4())[:8],
            started_at=datetime.now(timezone.utc).isoformat(),
            ended_at=None,
            peak_hazard=snapshot.hazard_level.value,
            fire_confirmed=snapshot.fire_state in (
                DetectionState.CONFIRMED, DetectionState.EMERGENCY
            ),
            smoke_confirmed=snapshot.smoke_state in (
                DetectionState.CONFIRMED, DetectionState.EMERGENCY
            ),
            max_people=snapshot.people_count,
            max_livestock=snapshot.livestock_count,
            peak_density=(
                snapshot.smoke_analysis.density_index
                if snapshot.smoke_analysis else 0.0
            ),
            frame_count=1,
        )
        self._save_event(event)
        return event

    def _update_event(self, snapshot: HazardSnapshot) -> None:
        e = self._active_event
        e.frame_count += 1
        if HazardLevel[snapshot.hazard_level.value].numeric > HazardLevel[e.peak_hazard].numeric:
            e.peak_hazard = snapshot.hazard_level.value
        e.fire_confirmed  = e.fire_confirmed  or snapshot.fire_state  in (DetectionState.CONFIRMED, DetectionState.EMERGENCY)
        e.smoke_confirmed = e.smoke_confirmed or snapshot.smoke_state in (DetectionState.CONFIRMED, DetectionState.EMERGENCY)
        e.max_people    = max(e.max_people,    snapshot.people_count)
        e.max_livestock = max(e.max_livestock, snapshot.livestock_count)
        if snapshot.smoke_analysis:
            e.peak_density = max(e.peak_density, snapshot.smoke_analysis.density_index)
        self._save_event(e)

    def _close_event(self) -> None:
        e = self._active_event
        e.ended_at = datetime.now(timezone.utc).isoformat()
        e.closed = True
        self._save_event(e)
        self._active_event = None

    def _save_event(self, event: FireGuardEvent) -> None:
        path = self._events_dir / f"{event.event_id}.json"
        path.write_text(json.dumps(event.to_dict(), indent=2))
