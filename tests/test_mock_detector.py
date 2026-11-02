"""
tests/test_mock_detector.py
────────────────────────────
Tests for the MockDetector — verifies the Detector interface contract.
"""

import numpy as np
import pytest
from core.detector import DetectionQuery
from models.locate_anything.mock_detector import MockDetector


@pytest.fixture
def detector():
    d = MockDetector(seed=0)
    d.load()
    yield d
    d.unload()


class TestMockDetector:

    def test_load_unload(self):
        d = MockDetector()
        assert not d.is_loaded()
        d.load()
        assert d.is_loaded()
        d.unload()
        assert not d.is_loaded()

    def test_detect_requires_load(self):
        d = MockDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError):
            d.detect(frame, DetectionQuery.FIRE)

    def test_detect_returns_detection(self, detector):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame, DetectionQuery.FIRE)
        assert result.query == DetectionQuery.FIRE
        assert isinstance(result.boxes, list)

    def test_boxes_in_range(self, detector):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for query in DetectionQuery:
            result = detector.detect(frame, query)
            for box in result.boxes:
                assert 0.0 <= box.x1 <= 1.0
                assert 0.0 <= box.y1 <= 1.0
                assert 0.0 <= box.x2 <= 1.0
                assert 0.0 <= box.y2 <= 1.0
                assert box.x1 < box.x2
                assert box.y1 < box.y2

    def test_vram_is_zero(self, detector):
        assert detector.vram_usage_mb == 0.0
