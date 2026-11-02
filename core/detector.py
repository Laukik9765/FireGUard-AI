"""
core/detector.py
────────────────
Detector abstraction layer.

FireGuard never talks to LocateAnything directly — all visual localization
requests go through this interface.  This means the underlying model can be
swapped out (e.g., fine-tuned weights, a different architecture) without
touching any other part of the application.

Usage
-----
    from core.detector import Detector, Detection, DetectionQuery

    detector: Detector = LocateAnythingDetector(config)
    results = detector.detect(image, DetectionQuery.FIRE)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Query types
# ──────────────────────────────────────────────────────────────────────────────

class DetectionQuery(str, Enum):
    """Natural-language detection targets supported by FireGuard."""
    FIRE      = "Locate visible fire"
    SMOKE     = "Locate visible smoke"
    PEOPLE    = "Locate all people"
    LIVESTOCK = "Locate all livestock and animals"

    def prompt(self) -> str:
        return self.value


# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """Normalised bounding box: x1, y1, x2, y2 in [0, 1] space."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    def to_pixel(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert normalised coords to pixel coords for a given frame size."""
        return (
            int(self.x1 * width),
            int(self.y1 * height),
            int(self.x2 * width),
            int(self.y2 * height),
        )

    @property
    def area_fraction(self) -> float:
        """Fraction of the frame covered by this box (0–1)."""
        return max(0.0, (self.x2 - self.x1) * (self.y2 - self.y1))


@dataclass
class Detection:
    """Single detection result returned by a Detector."""
    query: DetectionQuery
    boxes: list[BoundingBox] = field(default_factory=list)
    raw_output: Optional[dict] = field(default=None, repr=False)

    @property
    def count(self) -> int:
        return len(self.boxes)

    @property
    def detected(self) -> bool:
        return len(self.boxes) > 0

    @property
    def total_coverage(self) -> float:
        """Sum of all bounding-box area fractions (clamped to 1.0)."""
        return min(1.0, sum(b.area_fraction for b in self.boxes))


# ──────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ──────────────────────────────────────────────────────────────────────────────

class Detector(abc.ABC):
    """
    Abstract detector interface.

    All concrete detectors (LocateAnythingDetector, MockDetector, …) must
    implement this interface.
    """

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights into memory / GPU.  Called once at startup."""
        ...

    @abc.abstractmethod
    def unload(self) -> None:
        """Release model weights from GPU memory."""
        ...

    @abc.abstractmethod
    def detect(
        self,
        image: np.ndarray,         # BGR uint8, shape (H, W, 3)
        query: DetectionQuery,
        mode: str = "hybrid",      # "fast" | "slow" | "hybrid"
    ) -> Detection:
        """
        Run visual localization for the given query on a single frame.

        Parameters
        ----------
        image : np.ndarray
            BGR frame from OpenCV.
        query : DetectionQuery
            What to locate (fire, smoke, people, livestock).
        mode : str
            LocateAnything inference mode.

        Returns
        -------
        Detection
            Contains all bounding boxes found.
        """
        ...

    @abc.abstractmethod
    def is_loaded(self) -> bool:
        """Return True if model is currently resident in memory."""
        ...

    @property
    @abc.abstractmethod
    def vram_usage_mb(self) -> float:
        """Current GPU VRAM usage in MB (0 if not loaded / no GPU)."""
        ...
