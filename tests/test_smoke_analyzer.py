"""
tests/test_smoke_analyzer.py
─────────────────────────────
Unit tests for the SmokeAnalyzer module.
"""

import pytest
from core.detector import BoundingBox, Detection, DetectionQuery
from core.smoke_analyzer import SmokeAnalyzer, SmokeHazardLevel, SmokeTrend


def make_smoke_detection(boxes: list[tuple]) -> Detection:
    """Helper: create a smoke Detection from (x1,y1,x2,y2) tuples."""
    return Detection(
        query=DetectionQuery.SMOKE,
        boxes=[BoundingBox(*b) for b in boxes],
    )


def empty_smoke() -> Detection:
    return Detection(query=DetectionQuery.SMOKE, boxes=[])


class TestSmokeAnalyzer:

    def test_no_smoke_returns_none_level(self):
        sa = SmokeAnalyzer()
        result = sa.update(empty_smoke())
        assert not result.detected
        assert result.hazard_level == SmokeHazardLevel.NONE
        assert result.density_index == 0.0

    def test_small_smoke_returns_low_density(self):
        sa = SmokeAnalyzer()
        det = make_smoke_detection([(0.1, 0.1, 0.2, 0.2)])
        result = sa.update(det)
        assert result.detected
        assert result.density_index < 40  # small box → low score

    def test_large_smoke_returns_higher_density(self):
        sa = SmokeAnalyzer()
        det = make_smoke_detection([(0.0, 0.0, 0.8, 0.8)])
        result = sa.update(det)
        assert result.detected
        assert result.density_index > 40

    def test_trend_increases_over_time(self):
        sa = SmokeAnalyzer(history_len=6)
        # Feed small smoke, then large
        small = make_smoke_detection([(0.1, 0.1, 0.15, 0.15)])
        large = make_smoke_detection([(0.0, 0.0, 0.9, 0.9)])
        for _ in range(3):
            sa.update(small)
        for _ in range(3):
            sa.update(large)
        last = sa.update(large)
        assert last.trend == SmokeTrend.INCREASING

    def test_reset_clears_history(self):
        sa = SmokeAnalyzer()
        sa.update(make_smoke_detection([(0.0, 0.0, 0.9, 0.9)]))
        sa.reset()
        result = sa.update(empty_smoke())
        assert result.trend == SmokeTrend.UNKNOWN

    def test_coverage_pct_in_range(self):
        sa = SmokeAnalyzer()
        det = make_smoke_detection([(0.0, 0.0, 1.0, 1.0)])
        result = sa.update(det)
        assert 0.0 <= result.coverage_pct <= 100.0
