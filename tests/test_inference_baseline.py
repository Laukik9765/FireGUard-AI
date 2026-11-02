"""
tests/test_inference_baseline.py
──────────────────────────────────
Phase 1 baseline test: verify LocateAnything can run on the RTX 3050.

Run this FIRST before building the full pipeline.
DO NOT add to CI — this requires a GPU and model weights.

Usage
-----
    python tests/test_inference_baseline.py
"""

import logging
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("baseline_test")


def run_baseline():
    """Verify LocateAnything loads and produces output on the GPU."""
    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False

    if not torch.cuda.is_available():
        logger.error("CUDA not available. Cannot run baseline on CPU — model is too large.")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info("GPU: %s | VRAM: %.1f GB", gpu_name, vram_gb)

    if vram_gb < 4:
        logger.warning("Less than 4 GB VRAM detected — model may not load.")

    try:
        from models.locate_anything.locate_anything_detector import LocateAnythingDetector
        import numpy as np

        logger.info("Loading detector with 4-bit quantization …")
        t0 = time.perf_counter()

        detector = LocateAnythingDetector(quantization="4bit")
        detector.load()

        load_time = time.perf_counter() - t0
        logger.info("Loaded in %.1f s | VRAM: %.1f MB", load_time, detector.vram_usage_mb)

        from core.detector import DetectionQuery

        # Test on a blank frame first
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for query in DetectionQuery:
            t1 = time.perf_counter()
            result = detector.detect(frame, query)
            latency_ms = (time.perf_counter() - t1) * 1000
            logger.info(
                "Query: %-12s | Boxes: %d | Latency: %.1f ms",
                query.name, result.count, latency_ms,
            )

        detector.unload()
        logger.info("Baseline test PASSED.")
        return True

    except Exception as e:
        logger.error("Baseline test FAILED: %s", e)
        return False


if __name__ == "__main__":
    ok = run_baseline()
    sys.exit(0 if ok else 1)
