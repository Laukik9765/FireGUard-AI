"""
tests/test_hazard_engine.py
─────────────────────────────
Unit tests for the Temporal Hazard Engine.
"""

import pytest
from core.detector import BoundingBox, Detection, DetectionQuery
from core.hazard_engine import HazardEngine, HazardLevel, DetectionState


def fire_det(detected: bool) -> Detection:
    boxes = [BoundingBox(0.1, 0.1, 0.3, 0.3)] if detected else []
    return Detection(query=DetectionQuery.FIRE, boxes=boxes)


def empty(query: DetectionQuery) -> Detection:
    return Detection(query=query, boxes=[])


class TestHazardEngine:

    def test_no_detections_returns_normal(self):
        engine = HazardEngine()
        snap = engine.update(
            fire_detection=empty(DetectionQuery.FIRE),
            smoke_detection=empty(DetectionQuery.SMOKE),
            people_detection=empty(DetectionQuery.PEOPLE),
            livestock_detection=empty(DetectionQuery.LIVESTOCK),
        )
        assert snap.hazard_level == HazardLevel.NORMAL

    def test_single_fire_frame_does_not_confirm(self):
        """One fire frame should not immediately confirm fire (requires persistence)."""
        engine = HazardEngine(window=10, confirm_threshold=0.6)
        snap = engine.update(
            fire_detection=fire_det(True),
            smoke_detection=empty(DetectionQuery.SMOKE),
            people_detection=empty(DetectionQuery.PEOPLE),
            livestock_detection=empty(DetectionQuery.LIVESTOCK),
        )
        assert snap.fire_state != DetectionState.CONFIRMED

    def test_persistent_fire_escalates(self):
        """Multiple consecutive fire detections should escalate to CONFIRMED/EMERGENCY."""
        engine = HazardEngine(window=5, confirm_threshold=0.6)
        snap = None
        for _ in range(10):
            snap = engine.update(
                fire_detection=fire_det(True),
                smoke_detection=empty(DetectionQuery.SMOKE),
                people_detection=empty(DetectionQuery.PEOPLE),
                livestock_detection=empty(DetectionQuery.LIVESTOCK),
            )
        # Fire-only detections (no smoke, no people) correctly score MODERATE or above
        assert snap.fire_state in (DetectionState.CONFIRMED, DetectionState.EMERGENCY)
        assert snap.hazard_level.numeric >= HazardLevel.MODERATE.numeric

    def test_reset_clears_state(self):
        engine = HazardEngine(window=5, confirm_threshold=0.6)
        for _ in range(10):
            engine.update(
                fire_detection=fire_det(True),
                smoke_detection=empty(DetectionQuery.SMOKE),
                people_detection=empty(DetectionQuery.PEOPLE),
                livestock_detection=empty(DetectionQuery.LIVESTOCK),
            )
        engine.reset()
        snap = engine.update(
            fire_detection=empty(DetectionQuery.FIRE),
            smoke_detection=empty(DetectionQuery.SMOKE),
            people_detection=empty(DetectionQuery.PEOPLE),
            livestock_detection=empty(DetectionQuery.LIVESTOCK),
        )
        assert snap.hazard_level == HazardLevel.NORMAL
