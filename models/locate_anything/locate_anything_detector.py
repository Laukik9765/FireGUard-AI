"""
models/locate_anything/locate_anything_detector.py
────────────────────────────────────────────────────
Concrete implementation of the Detector interface using NVIDIA LocateAnything-3B.

This module is the ONLY place in the codebase that imports or references
LocateAnything directly.  Everything else uses the abstract Detector interface.

Hardware target: RTX 3050 (6 GB VRAM) — uses 4-bit quantization by default.

Setup notes
-----------
1.  Install PyTorch with CUDA:
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
2.  Install HuggingFace Transformers >= 4.40 and bitsandbytes >= 0.43.
3.  Download model weights (first run will auto-download via HuggingFace Hub):
        nvidia/locate-anything-3b   (check NVIDIA NGC / HuggingFace for the
                                     exact repo name when available)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch
from PIL import Image

from core.detector import BoundingBox, Detection, DetectionQuery, Detector

logger = logging.getLogger(__name__)

# ─── Default model identifier ─────────────────────────────────────────────────
# Replace with the verified HuggingFace / NGC model ID when the model is
# publicly released.  See README for current status.
DEFAULT_MODEL_ID = "nvidia/locate-anything-3b"   # placeholder — verify before use

# ─── Quantization presets ─────────────────────────────────────────────────────
# Ordered from most to least aggressive — fall back if VRAM is insufficient.
QUANTIZATION_PRESETS = ["4bit", "8bit", "none"]


class LocateAnythingDetector(Detector):
    """
    Wraps NVIDIA LocateAnything-3B behind the Detector interface.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    quantization : str
        "4bit" | "8bit" | "none"
    device : str
        "cuda" | "cpu"
    inference_mode : str
        Default inference mode: "fast" | "slow" | "hybrid"
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        quantization: str = "4bit",
        device: str = "cuda",
        inference_mode: str = "hybrid",
    ) -> None:
        self._model_id = model_id
        self._quantization = quantization
        self._device = device if torch.cuda.is_available() else "cpu"
        self._default_mode = inference_mode

        self._model = None
        self._processor = None
        self._loaded = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load LocateAnything-3B into GPU memory with quantization."""
        if self._loaded:
            logger.warning("Detector already loaded — skipping.")
            return

        logger.info(
            "Loading LocateAnything [%s] with %s quantization on %s …",
            self._model_id, self._quantization, self._device,
        )

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

            bnb_config = None
            if self._quantization == "4bit" and self._device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif self._quantization == "8bit" and self._device == "cuda":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                quantization_config=bnb_config,
                device_map="auto" if self._device == "cuda" else None,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                trust_remote_code=True,
            )

            self._processor = AutoProcessor.from_pretrained(
                self._model_id,
                trust_remote_code=True,
            )

            self._loaded = True
            logger.info(
                "LocateAnything loaded.  VRAM usage: %.1f MB", self.vram_usage_mb
            )

        except ImportError as e:
            raise RuntimeError(
                "Required library missing.  Run: pip install transformers bitsandbytes"
            ) from e
        except Exception as e:
            logger.error("Failed to load LocateAnything: %s", e)
            raise

    def unload(self) -> None:
        """Release model from GPU memory."""
        if not self._loaded:
            return
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("LocateAnything unloaded.")

    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def vram_usage_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / 1024 / 1024

    # ── Inference ─────────────────────────────────────────────────────────────

    def detect(
        self,
        image: np.ndarray,
        query: DetectionQuery,
        mode: str = "hybrid",
    ) -> Detection:
        """
        Run LocateAnything inference for the given query.

        The image is expected in BGR format (from OpenCV).  It is converted to
        RGB/PIL internally before being passed to the model processor.
        """
        if not self._loaded:
            raise RuntimeError("Detector not loaded.  Call detector.load() first.")

        # BGR → RGB → PIL
        rgb_image = Image.fromarray(image[:, :, ::-1])
        h, w = image.shape[:2]

        prompt = query.prompt()
        t0 = time.perf_counter()

        try:
            # NOTE: The exact API depends on the LocateAnything release.
            # Update the processor call below when the official API is confirmed.
            inputs = self._processor(
                images=rgb_image,
                text=prompt,
                return_tensors="pt",
            )
            if self._device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                )

            # Decode output — the format depends on the LocateAnything API.
            # This is a placeholder; update when the actual output schema is known.
            decoded = self._processor.decode(outputs[0], skip_special_tokens=True)
            boxes = self._parse_boxes(decoded, w, h)

        except Exception as e:
            logger.error("Inference failed for query '%s': %s", prompt, e)
            boxes = []
            decoded = ""

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "detect(%s) → %d boxes in %.1f ms", query.name, len(boxes), latency_ms
        )

        return Detection(
            query=query,
            boxes=boxes,
            raw_output={"decoded": decoded, "latency_ms": latency_ms},
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_boxes(
        self, decoded_text: str, frame_w: int, frame_h: int
    ) -> list[BoundingBox]:
        """
        Parse bounding box coordinates from the model's decoded output.

        ⚠️  This is a PLACEHOLDER implementation.  The actual output format
        of LocateAnything must be confirmed from official docs/examples and
        this parser updated accordingly.

        LocateAnything likely outputs coordinates normalised to [0, 1] or as
        pixel integers — verify and update before use.
        """
        import re

        boxes = []
        # Example pattern: [x1, y1, x2, y2] repeated — update to match actual format
        pattern = r"\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]"
        matches = re.findall(pattern, decoded_text)

        for m in matches:
            x1, y1, x2, y2 = map(float, m)
            # Normalise if values look like pixels (> 1.0)
            if x2 > 1.0 or y2 > 1.0:
                x1 /= frame_w
                y1 /= frame_h
                x2 /= frame_w
                y2 /= frame_h
            x1, y1, x2, y2 = (
                max(0.0, min(1.0, v)) for v in (x1, y1, x2, y2)
            )
            boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))

        return boxes
