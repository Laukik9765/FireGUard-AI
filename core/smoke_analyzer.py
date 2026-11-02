"""
core/smoke_analyzer.py
────────────────────────
FireGuard-specific smoke analysis layer.

Raw LocateAnything smoke detections are NOT treated directly as danger metrics.
This module computes:
  - Smoke Coverage   (% of visible scene)
  - Smoke Density Index  (0–100 internal scale)
  - Smoke Hazard Level   (None / Low / Moderate / High / Critical)
  - Smoke Trend          (Increasing / Stable / Decreasing)

⚠️  The Density Index is an internal engineering indicator ONLY.
     It must never be presented as an official fire-safety standard.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from core.detector import Detection, DetectionQuery


# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────

class SmokeHazardLevel(str, Enum):
    NONE     = "None"
    VERY_LOW = "Very Low"
    LOW      = "Low"
    MODERATE = "Moderate"
    HIGH     = "High"
    CRITICAL = "Critical"

    @property
    def color_key(self) -> str:
        """CSS color token name for the dashboard."""
        return {
            "None":     "color-safe",
            "Very Low": "color-safe",
            "Low":      "color-warn-low",
            "Moderate": "color-warn-moderate",
            "High":     "color-warn-high",
            "Critical": "color-critical",
        }[self.value]


class SmokeTrend(str, Enum):
    INCREASING = "Increasing"
    STABLE     = "Stable"
    DECREASING = "Decreasing"
    UNKNOWN    = "Unknown"

    @property
    def arrow(self) -> str:
        return {
            "Increasing": "↑",
            "Stable":     "→",
            "Decreasing": "↓",
            "Unknown":    "—",
        }[self.value]


@dataclass
class SmokeAnalysis:
    """Complete smoke analysis result for one frame."""
    coverage_pct: float           # 0–100: % of scene visually covered
    density_index: float          # 0–100: internal indicator (NOT an official standard)
    hazard_level: SmokeHazardLevel
    trend: SmokeTrend
    detected: bool
    box_count: int

    def __str__(self) -> str:
        if not self.detected:
            return "SMOKE: None"
        return (
            f"SMOKE: {self.hazard_level.value} | "
            f"Density: {self.density_index:.0f}/100 | "
            f"Coverage: {self.coverage_pct:.0f}% | "
            f"Trend: {self.trend.arrow} {self.trend.value}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Density table
# ──────────────────────────────────────────────────────────────────────────────

def _density_to_level(density: float) -> SmokeHazardLevel:
    """
    Map a density index (0–100) to a hazard level.

    Thresholds are initial engineering values — they must be calibrated with
    real FireGuard-specific footage before deployment.
    """
    if density <= 0:
        return SmokeHazardLevel.NONE
    if density <= 20:
        return SmokeHazardLevel.VERY_LOW
    if density <= 40:
        return SmokeHazardLevel.LOW
    if density <= 60:
        return SmokeHazardLevel.MODERATE
    if density <= 80:
        return SmokeHazardLevel.HIGH
    return SmokeHazardLevel.CRITICAL


# ──────────────────────────────────────────────────────────────────────────────
# Smoke Analyzer
# ──────────────────────────────────────────────────────────────────────────────

class SmokeAnalyzer:
    """
    Computes smoke density, coverage, and trend from a sequence of detections.

    Parameters
    ----------
    history_len : int
        Number of past frames to consider for trend analysis.
    coverage_weight : float
        How much raw coverage contributes to the density index (0–1).
    persistence_weight : float
        How much frame-persistence contributes to the density index (0–1).
    """

    def __init__(
        self,
        history_len: int = 10,
        coverage_weight: float = 0.6,
        persistence_weight: float = 0.4,
    ) -> None:
        if abs(coverage_weight + persistence_weight - 1.0) > 1e-6:
            raise ValueError("coverage_weight + persistence_weight must equal 1.0")

        self._history_len = history_len
        self._coverage_weight = coverage_weight
        self._persistence_weight = persistence_weight

        # Ring-buffer of (coverage_pct, density_index) tuples
        self._history: deque[float] = deque(maxlen=history_len)
        self._consecutive_detections: int = 0

    def update(self, detection: Detection) -> SmokeAnalysis:
        """
        Feed a new smoke detection result and compute the analysis.

        Parameters
        ----------
        detection : Detection
            Must be the SMOKE query result.

        Returns
        -------
        SmokeAnalysis
        """
        assert detection.query == DetectionQuery.SMOKE, (
            "SmokeAnalyzer.update() expects a SMOKE detection."
        )

        if not detection.detected:
            self._consecutive_detections = max(0, self._consecutive_detections - 1)
            self._history.append(0.0)
            trend = self._compute_trend()
            return SmokeAnalysis(
                coverage_pct=0.0,
                density_index=0.0,
                hazard_level=SmokeHazardLevel.NONE,
                trend=trend,
                detected=False,
                box_count=0,
            )

        self._consecutive_detections += 1

        # ── Coverage ───────────────────────────────────────────────────────────
        # Sum of all bounding-box areas as a fraction of the full frame (capped)
        coverage_fraction = min(1.0, detection.total_coverage)
        coverage_pct = coverage_fraction * 100.0

        # ── Persistence component ─────────────────────────────────────────────
        # The longer smoke has been continuously present, the higher the score
        persistence_score = min(1.0, self._consecutive_detections / self._history_len)

        # ── Density index ─────────────────────────────────────────────────────
        density_index = (
            coverage_fraction * self._coverage_weight
            + persistence_score * self._persistence_weight
        ) * 100.0

        self._history.append(density_index)
        trend = self._compute_trend()

        return SmokeAnalysis(
            coverage_pct=round(coverage_pct, 1),
            density_index=round(density_index, 1),
            hazard_level=_density_to_level(density_index),
            trend=trend,
            detected=True,
            box_count=detection.count,
        )

    def reset(self) -> None:
        """Clear all history (e.g., when switching camera feeds)."""
        self._history.clear()
        self._consecutive_detections = 0

    # ── Private ───────────────────────────────────────────────────────────────

    def _compute_trend(self) -> SmokeTrend:
        if len(self._history) < 3:
            return SmokeTrend.UNKNOWN

        recent = list(self._history)
        half = len(recent) // 2
        old_avg = sum(recent[:half]) / half
        new_avg = sum(recent[half:]) / (len(recent) - half)
        delta = new_avg - old_avg

        if delta > 5:
            return SmokeTrend.INCREASING
        if delta < -5:
            return SmokeTrend.DECREASING
        return SmokeTrend.STABLE
