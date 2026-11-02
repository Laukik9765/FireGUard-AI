"""
tests/test_false_positive_rejection.py
───────────────────────────────────────
Verifies that MockDetector does not trigger false positive fire/smoke alarms
on non-fire red/orange objects, plain backgrounds, and noise.
"""

import cv2
import numpy as np
import pytest

from core.detector import DetectionQuery
from models.locate_anything.mock_detector import MockDetector


@pytest.fixture
def detector():
    d = MockDetector()
    d.load()
    yield d
    d.unload()


class TestFalsePositiveRejection:

    def test_black_frame_zero_detections(self, detector):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fire = detector.detect(frame, DetectionQuery.FIRE)
        smoke = detector.detect(frame, DetectionQuery.SMOKE)
        people = detector.detect(frame, DetectionQuery.PEOPLE)

        assert len(fire.boxes) == 0, "Black frame should have 0 fire detections"
        assert len(smoke.boxes) == 0, "Black frame should have 0 smoke detections"
        assert len(people.boxes) == 0, "Black frame should have 0 people detections"

    def test_solid_red_object_rejected(self, detector):
        """A uniform solid red box (like a red mug, t-shirt, or poster) must NOT trigger fire."""
        frame = np.full((480, 640, 3), [100, 100, 100], dtype=np.uint8)  # grey background
        # Draw solid red rectangle (flat color, low variance)
        cv2.rectangle(frame, (200, 150), (440, 350), (0, 0, 230), -1)

        # Run across multiple frames to test temporal tracker
        for _ in range(5):
            fire = detector.detect(frame, DetectionQuery.FIRE)

        assert len(fire.boxes) == 0, "Flat solid red object must not trigger false positive fire detection"

    def test_sunset_orange_gradient_rejected(self, detector):
        """Warm sunset or room lighting gradient must not trigger fire."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for y in range(480):
            frame[y, :] = [30, int(80 + 40 * (y / 480)), int(160 + 60 * (y / 480))]

        for _ in range(5):
            fire = detector.detect(frame, DetectionQuery.FIRE)

        assert len(fire.boxes) == 0, "Warm lighting gradient must not trigger false positive fire"

    def test_white_wall_rejected_for_smoke(self, detector):
        """A plain painted white/light grey wall must not trigger smoke."""
        frame = np.full((480, 640, 3), [220, 220, 220], dtype=np.uint8)
        # Add high edge density grid (e.g. wall texture)
        for x in range(0, 640, 40):
            cv2.line(frame, (x, 0), (x, 480), (180, 180, 180), 2)

        for _ in range(5):
            smoke = detector.detect(frame, DetectionQuery.SMOKE)

        assert len(smoke.boxes) == 0, "Wall texture must not trigger smoke"
