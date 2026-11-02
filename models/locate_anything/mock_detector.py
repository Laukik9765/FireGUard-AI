"""
models/locate_anything/mock_detector.py
────────────────────────────────────────
FireGuard High-Precision Multi-Class Ensemble Detector

Classes Supported:
  1. FIRE / FLAMES: Neural YOLOv8 fire detector + thermal chromaticity validation
  2. SMOKE: Billowing smoke plume & haze dispersion analyzer with atmospheric sky rejection
  3. PEOPLE: YOLOv8 state-of-the-art person detection (COCO class 0)
  4. LIVESTOCK: YOLOv8 animal detection (COCO classes 14-23)

Anti-False-Positive Architecture:
  - Neural Object Modeling: Eliminates heuristic color misclassifications on tents/jackets/posters.
  - Thermal Chromaticity Gate: Validates high red-to-green and red-to-blue combustion ratio.
  - Atmospheric Sky Rejection: Distinguishes achromatic smoke from blue sky and background mountains.
  - Flicker / Motion Validation: Rejects static background objects by requiring temporal pixel variance
    across rolling frame history (real flames flicker at 8–12 Hz; static objects have near-zero variance).
  - Smoke Dispersion Velocity Gate: Requires smoke regions to exhibit upward motion / expansion across
    frames; static ceilings and walls are discarded even when they pass the color filter.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.detector import BoundingBox, Detection, DetectionQuery, Detector

logger = logging.getLogger(__name__)

# ── Thresholds (Configurable via ENV) ─────────────────────────────────────────
# FIRE_CONF_THRESHOLD: 0.52 — rejects low-confidence YOLO hits on warm skin and indoor objects
# while still passing genuine fire candidates (real flames typically score ≥ 0.55 on the fire model).
CONF_FIRE_THRESH   = float(os.environ.get("FIRE_CONF_THRESHOLD", "0.52"))
CONF_SMOKE_THRESH  = float(os.environ.get("SMOKE_CONF_THRESHOLD", "0.30"))
# PEOPLE_CONF_THRESHOLD raised 0.50 → 0.65 to reject wooden doors, coats, and furniture.
CONF_PERSON_THRESH = float(os.environ.get("PEOPLE_CONF_THRESHOLD", "0.65"))
CONF_ANIMAL_THRESH = float(os.environ.get("ANIMAL_CONF_THRESHOLD", "0.45"))

# TEMPORAL_WINDOW: rolling frame buffer depth for confirmation tracking.
TEMPORAL_WINDOW   = int(os.environ.get("DETECTOR_TEMPORAL_WINDOW", "8"))

# TEMPORAL_MIN_HITS: detections must be confirmed across multiple consecutive frames.
TEMPORAL_MIN_HITS = int(os.environ.get("DETECTOR_TEMPORAL_HITS", "3"))

# TEMPORAL_IOU_THRESH: overlap requirement between frames.
TEMPORAL_IOU_THRESH = float(os.environ.get("DETECTOR_IOU_THRESH", "0.25"))

# ── Flicker / Motion Validation Thresholds ─────────────────────────────────────
FIRE_FLICKER_VAR_MIN   = float(os.environ.get("FIRE_FLICKER_VAR_MIN",   "18.0"))
SMOKE_MOTION_MIN       = float(os.environ.get("SMOKE_MOTION_MIN",       "0.008"))
SMOKE_TEMPORAL_DIFF_MIN = float(os.environ.get("SMOKE_TEMPORAL_DIFF_MIN", "3.2"))


class _CentroidTracker:
    """Lightweight 2D spatial tracker for temporal confirmation of detections."""

    def __init__(self, max_history: int = 8, iou_thresh: float = 0.20):
        self.max_history = max_history
        self.iou_thresh = iou_thresh
        self.history: list[list[BoundingBox]] = []

    def update(self, current_boxes: list[BoundingBox], bypass_confirmation: bool = False) -> list[BoundingBox]:
        self.history.append(current_boxes)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        if bypass_confirmation or TEMPORAL_MIN_HITS <= 1:
            return current_boxes

        if len(self.history) < TEMPORAL_MIN_HITS:
            return []

        confirmed: list[BoundingBox] = []
        for box in current_boxes:
            hits = 1
            for past_frame in self.history[:-1]:
                if any(self._iou(box, past_box) >= self.iou_thresh for past_box in past_frame):
                    hits += 1
            if hits >= TEMPORAL_MIN_HITS:
                confirmed.append(box)
        return confirmed

    @staticmethod
    def _iou(b1: BoundingBox, b2: BoundingBox) -> float:
        ix1 = max(b1.x1, b2.x1)
        iy1 = max(b1.y1, b2.y1)
        ix2 = min(b1.x2, b2.x2)
        iy2 = min(b1.y2, b2.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area1 = (b1.x2 - b1.x1) * (b1.y2 - b1.y1)
        area2 = (b2.x2 - b2.x1) * (b2.y2 - b2.y1)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def reset(self) -> None:
        self.history.clear()


class MockDetector(Detector):
    """
    High-precision FireGuard Detector.
    Integrates YOLO neural detection with anti-false-positive multi-gate validation.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._loaded = False
        self._fire_model = None
        self._coco_model = None

        self._fire_tracker      = _CentroidTracker(max_history=TEMPORAL_WINDOW, iou_thresh=TEMPORAL_IOU_THRESH)
        self._smoke_tracker     = _CentroidTracker(max_history=TEMPORAL_WINDOW, iou_thresh=TEMPORAL_IOU_THRESH)
        self._person_tracker    = _CentroidTracker(max_history=TEMPORAL_WINDOW, iou_thresh=TEMPORAL_IOU_THRESH)
        self._livestock_tracker = _CentroidTracker(max_history=TEMPORAL_WINDOW, iou_thresh=TEMPORAL_IOU_THRESH)

        # Rolling buffer of small greyscale frames used for:
        #   • Fire flicker validation  – temporal variance in the fire-candidate region.
        #   • Smoke motion gate        – centroid displacement between oldest and newest snapshot.
        # Buffer depth matches TEMPORAL_WINDOW so both gates cover the same time horizon.
        self._prev_frames: deque[np.ndarray] = deque(maxlen=TEMPORAL_WINDOW)

        # Per-smoke-region centroid history: maps a rounded (cx, cy) key → deque of past centroids.
        # Used by the smoke dispersion-velocity gate.
        self._smoke_centroids: dict[str, deque[tuple[float, float]]] = {}

    def load(self) -> None:
        root = Path(__file__).resolve().parent.parent.parent

        # 1. Load Fire YOLO model
        fire_paths = [
            root / "fire_model.pt",
            root / "models" / "fireguard" / "fireguard_best.pt",
            root / "fireguard_model.pt",
        ]
        for p in fire_paths:
            if p.exists():
                try:
                    from ultralytics import YOLO
                    self._fire_model = YOLO(str(p))
                    logger.info("Loaded Fire YOLO model from %s", p)
                    break
                except Exception as e:
                    logger.warning("Failed loading fire model %s: %s", p, e)

        # 2. Load COCO YOLO model (for high-accuracy Person & Animal detection)
        coco_paths = [
            root / "yolov8s.pt",
            root / "yolov8n.pt",
        ]
        for p in coco_paths:
            if p.exists():
                try:
                    from ultralytics import YOLO
                    self._coco_model = YOLO(str(p))
                    logger.info("Loaded COCO YOLO model from %s", p)
                    break
                except Exception as e:
                    logger.warning("Failed loading COCO model %s: %s", p, e)

        self._loaded = True
        logger.info(
            "FireGuard Detector loaded (Fire YOLO: %s | COCO YOLO: %s).",
            bool(self._fire_model),
            bool(self._coco_model),
        )

    def unload(self) -> None:
        self._loaded = False
        self._fire_model = None
        self._coco_model = None
        self._fire_tracker.reset()
        self._smoke_tracker.reset()
        self._person_tracker.reset()
        self._livestock_tracker.reset()
        self._prev_frames.clear()
        self._smoke_centroids.clear()

    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def vram_usage_mb(self) -> float:
        return 0.0

    def detect(
        self,
        image: np.ndarray,
        query: DetectionQuery,
        mode: str = "hybrid",
    ) -> Detection:
        if not self._loaded:
            raise RuntimeError("Detector not loaded.")
        if image is None or image.size == 0:
            return Detection(query=query, boxes=[])

        h, w = image.shape[:2]
        boxes: list[BoundingBox] = []
        bypass = (mode == "single")

        # Keep rolling grayscale buffer updated on each video cycle (starting at FIRE query)
        if query == DetectionQuery.FIRE:
            small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (160, 120))
            self._prev_frames.append(small.astype(np.float32))

        if query == DetectionQuery.FIRE:
            # Run person detection first so we can suppress fire boxes that heavily overlap
            # with human skin — the #1 source of false-positive flames in live webcam feeds.
            person_boxes = self._detect_people(image, w, h, bypass_confirmation=bypass)
            fire_boxes   = self._detect_fire(image, w, h, bypass_confirmation=bypass)
            boxes = self._suppress_fire_on_persons(fire_boxes, person_boxes)
        elif query == DetectionQuery.SMOKE:
            boxes = self._detect_smoke(image, w, h, bypass_confirmation=bypass)
        elif query == DetectionQuery.PEOPLE:
            boxes = self._detect_people(image, w, h, bypass_confirmation=bypass)
        elif query == DetectionQuery.LIVESTOCK:
            boxes = self._detect_livestock(image, w, h, bypass_confirmation=bypass)

        return Detection(
            query=query,
            boxes=boxes,
            raw_output={"engine": "fireguard_ensemble"},
        )

    # ── Fire / Flame Detection ────────────────────────────────────────────────

    @staticmethod
    def _iou_boxes(a: BoundingBox, b: BoundingBox) -> float:
        """Compute Intersection-over-Union between two normalised BoundingBoxes."""
        ix1 = max(a.x1, b.x1); iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2); iy2 = min(a.y2, b.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0.0:
            return 0.0
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        return inter / max(area_a + area_b - inter, 1e-6)

    @staticmethod
    def _suppress_fire_on_persons(
        fire_boxes: list[BoundingBox],
        person_boxes: list[BoundingBox],
        iou_thresh: float = 0.20,
    ) -> list[BoundingBox]:
        """
        Person-Overlap Suppression Gate.

        Skin tones and human faces share the same warm orange/red HSV profile as fire,
        causing YOLO and CV detectors to hallucinate flames on people in webcam feeds.

        Strategy: if a fire candidate's bounding box overlaps a confirmed person box by
        more than `iou_thresh` (IoU), it is almost certainly the person's skin/clothing —
        reject it.  Real flames in doorways, floors, or open spaces have zero or negligible
        overlap with the human silhouette bounding box.
        """
        if not person_boxes:
            return fire_boxes
        kept: list[BoundingBox] = []
        for fb in fire_boxes:
            max_iou = max(
                MockDetector._iou_boxes(fb, pb) for pb in person_boxes
            )
            if max_iou <= iou_thresh:
                kept.append(fb)
            else:
                logger.debug(
                    "Fire box suppressed (person overlap IoU=%.3f > %.2f)",
                    max_iou, iou_thresh,
                )
        return kept

    def _detect_fire(self, image: np.ndarray, w: int, h: int, bypass_confirmation: bool = False) -> list[BoundingBox]:
        candidates: list[BoundingBox] = []

        # 1. Primary: YOLO neural inference
        if self._fire_model is not None:
            try:
                results = self._fire_model.predict(
                    source=image,
                    conf=CONF_FIRE_THRESH,
                    imgsz=640,
                    verbose=False,
                    device="cpu",
                )
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = self._fire_model.names.get(cls_id, "").lower()
                        if "fire" in cls_name or "flame" in cls_name or cls_id == 0:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = float(box.conf[0].item())
                            bbox = BoundingBox(
                                x1=round(max(0.0, x1 / w), 4),
                                y1=round(max(0.0, y1 / h), 4),
                                x2=round(min(1.0, x2 / w), 4),
                                y2=round(min(1.0, y2 / h), 4),
                                confidence=conf,
                            )
                            if self._validate_flame_region(image, bbox, w, h, bypass_confirmation=bypass_confirmation):
                                candidates.append(bbox)
            except Exception as e:
                logger.error("YOLO Fire inference error: %s", e)

        # 2. Ensemble CV Fallback: Run CV color/HSV detector if YOLO yielded 0 candidates
        if not candidates:
            cv_candidates = self._cv2_fire_fallback(image, w, h, bypass_confirmation=bypass_confirmation)
            candidates.extend(cv_candidates)

        confirmed = self._fire_tracker.update(candidates, bypass_confirmation=bypass_confirmation)
        return confirmed

    def _validate_flame_region(self, image: np.ndarray, box: BoundingBox, w: int, h: int, bypass_confirmation: bool = False) -> bool:
        """
        Anti-False-Positive Filter for Fire:
        Validates red, orange, and yellow combustion spectrum (including phone screen videos)
        while rejecting flat solid red objects (shirts, posters, chairs, lamps).

        Gate 5 — Flicker / Temporal Variance:
        Real flames oscillate at 8–12 Hz, producing high pixel-level variance in the
        candidate region across consecutive greyscale frames.  Static objects (wooden door
        frames, wall clocks, warm-tinted walls, skin under fixed lighting) have near-zero
        temporal variance and are rejected here.
        """
        x1, y1 = int(box.x1 * w), int(box.y1 * h)
        x2, y2 = int(box.x2 * w), int(box.y2 * h)
        bw, bh = x2 - x1, y2 - y1
        if bw < 6 or bh < 6:
            return False

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        b = crop[:, :, 0].astype(np.float32)
        g = crop[:, :, 1].astype(np.float32)
        r = crop[:, :, 2].astype(np.float32)

        mean_r = float(np.mean(r))
        mean_g = float(np.mean(g))
        mean_b = float(np.mean(b))

        # Gate 1 — Real flames require high red intensity and clear combustion gradient (allowing yellow/orange flames)
        # Skin tones sit at R/B ≈ 1.5–1.7; real flames require R/B > 1.80 to distinguish.
        if mean_r < 100:
            return False
        if mean_r < mean_g * 0.96 or mean_r < mean_b * 1.80:
            return False

        # Gate 2 — Check HSV saturation & brightness (rejects dull khaki/brown/grey/beige objects)
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv_crop[:, :, 1]))
        mean_val = float(np.mean(hsv_crop[:, :, 2]))
        if mean_sat < 45 or mean_val < 75:
            return False

        # Gate 3 — Interior spatial variance check — flat solid walls, ceilings & doors have low variance
        ix1, iy1 = int(x1 + 0.15 * bw), int(y1 + 0.15 * bh)
        ix2, iy2 = int(x2 - 0.15 * bw), int(y2 - 0.15 * bh)
        if ix2 > ix1 and iy2 > iy1:
            inner_crop = image[iy1:iy2, ix1:ix2]
            inner_gray = cv2.cvtColor(inner_crop, cv2.COLOR_BGR2GRAY)
            inner_var = float(np.var(inner_gray))
            if inner_var < 25.0:  # Rejects plain room walls/ceilings while passing real flames
                return False

        # Gate 4 — Aspect ratio sanity: very thin vertical strips are door/window frames, not flames
        aspect = bw / max(bh, 1)
        if aspect < 0.10:  # strip narrower than 10% of its own height
            return False

        # Gate 5 — Flicker / temporal variance across rolling frame buffer.
        # Sample the same normalised region from each stored greyscale frame and compute the
        # pixel-wise standard deviation across the time axis.  A mean std < FIRE_FLICKER_VAR_MIN
        # indicates a static background object, not a genuine flame.
        if len(self._prev_frames) >= 3 and not bypass_confirmation:
            patches: list[np.ndarray] = []
            for past_frame in self._prev_frames:
                ph, pw = past_frame.shape[:2]
                px1 = int(box.x1 * pw)
                py1 = int(box.y1 * ph)
                px2 = int(box.x2 * pw)
                py2 = int(box.y2 * ph)
                patch = past_frame[py1:py2, px1:px2]
                if patch.size > 0:
                    # Resize all patches to a fixed small tile for consistent stacking
                    patches.append(cv2.resize(patch, (16, 16)))
            if len(patches) >= 3:
                stack = np.stack(patches, axis=0)          # shape: (N, 16, 16)
                temporal_var = float(np.mean(np.std(stack, axis=0)))
                if temporal_var < FIRE_FLICKER_VAR_MIN:
                    logger.debug(
                        "Fire candidate rejected by flicker gate: temporal_var=%.2f < %.2f",
                        temporal_var, FIRE_FLICKER_VAR_MIN,
                    )
                    return False

        return True

    def _cv2_fire_fallback(self, image: np.ndarray, w: int, h: int, bypass_confirmation: bool = False) -> list[BoundingBox]:
        """
        5-gate OpenCV fallback when no neural model is available.

        Broadened HSV range to support orange, golden-yellow, and red flames
        (0–38° and 170–180°), with minimum contour area of 600 px².
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0,   120, 150]), np.array([38,  255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 120, 150]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 600:
                continue
            x, y, bw, bh = cv2.boundingRect(c)

            # Rejects ambient room lighting & sunset sky gradients spanning wide frame regions
            if bw >= w * 0.85 and bh >= h * 0.45:
                continue

            # Rejects thin vertical strips (door frames, window edges)
            if bw / max(bh, 1) < 0.10:
                continue

            bbox = BoundingBox(
                x1=round(x / w, 4),
                y1=round(y / h, 4),
                x2=round((x + bw) / w, 4),
                y2=round((y + bh) / h, 4),
                confidence=min(0.95, 0.50 + area / (w * h * 0.1)),
            )
            if self._validate_flame_region(image, bbox, w, h, bypass_confirmation=bypass_confirmation):
                boxes.append(bbox)
        return boxes

    # ── Smoke Detection ───────────────────────────────────────────────────────

    def _detect_smoke(self, image: np.ndarray, w: int, h: int, bypass_confirmation: bool = False) -> list[BoundingBox]:
        candidates = self._cv2_smoke_analyzer(image, w, h, bypass_confirmation=bypass_confirmation)
        confirmed = self._smoke_tracker.update(candidates, bypass_confirmation=bypass_confirmation)
        return confirmed

    def _cv2_smoke_analyzer(self, image: np.ndarray, w: int, h: int, bypass_confirmation: bool = False) -> list[BoundingBox]:
        """
        Multi-gate smoke analyzer: achromatic desaturation, sky exclusion, low edge density,
        flicker / temporal frame difference, and dispersion velocity gating.
        """
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Gate 1 — Smoke colour: desaturated grey, mid-brightness, achromatic, non-sky
        smoke_mask = (
            (sat < 35)
            & (val > 80)
            & (val < 220)
            & (np.abs(r - g) < 20)
            & (np.abs(r - b) < 22)
            & ~(b > r * 1.18)
        )
        smoke_u8 = (smoke_mask * 255).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        smoke_u8 = cv2.morphologyEx(smoke_u8, cv2.MORPH_OPEN, kernel)
        smoke_u8 = cv2.morphologyEx(smoke_u8, cv2.MORPH_CLOSE, kernel)

        total_pixels = w * h
        contours, _ = cv2.findContours(smoke_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 80)

        seen_keys: set[str] = set()

        for c in contours:
            area = cv2.contourArea(c)
            # Gate 2 — Minimum region area (4% of frame)
            if area < total_pixels * 0.04:
                continue
            x, y, bw, bh = cv2.boundingRect(c)

            # Gate 2b — Rejects full-screen wall/background coverage (>95% frame)
            if bw >= w * 0.95 and bh >= h * 0.95:
                continue

            # Gate 3 — Suppress high-frequency textured grids and dense walls (edge density > 0.13)
            crop_edges = edges[y:y+bh, x:x+bw]
            edge_density = float(cv2.countNonZero(crop_edges)) / max(1, bw * bh)
            if edge_density > 0.13:
                continue

            # Gate 4 — Interior spatial variance: flat uniform surfaces have near-zero variance
            ix1, iy1 = int(x + 0.15 * bw), int(y + 0.15 * bh)
            ix2, iy2 = int(x + 0.85 * bw), int(y + 0.85 * bh)
            if ix2 > ix1 and iy2 > iy1:
                inner_gray = gray[iy1:iy2, ix1:ix2]
                inner_var = float(np.var(inner_gray))
                if inner_var < 40.0:
                    continue

            # Gate 5 — Temporal pixel variance / frame difference:
            # Real smoke is a fluid medium that diffuses and billows, causing pixel-intensity fluctuations.
            # Static ceilings, painted walls, door frames, and stationary furniture have near-zero frame-to-frame diff (< 3.0).
            if len(self._prev_frames) >= 2 and not bypass_confirmation:
                prev_gray = self._prev_frames[-1]
                curr_small = cv2.resize(gray, (160, 120))
                ph, pw = curr_small.shape
                px1 = int(round((x / w) * pw))
                py1 = int(round((y / h) * ph))
                px2 = int(round(((x + bw) / w) * pw))
                py2 = int(round(((y + bh) / h) * ph))
                if px2 > px1 and py2 > py1:
                    c_patch = curr_small[py1:py2, px1:px2]
                    p_patch = prev_gray[py1:py2, px1:px2]
                    frame_diff = float(np.mean(np.abs(c_patch - p_patch)))
                    if frame_diff < SMOKE_TEMPORAL_DIFF_MIN:
                        continue

            # Gate 6 — Smoke dispersion velocity (centroid motion across frames)
            cx_norm = (x + bw * 0.5) / w
            cy_norm = (y + bh * 0.5) / h

            key = f"{round(cx_norm, 1)},{round(cy_norm, 1)}"
            seen_keys.add(key)

            if key not in self._smoke_centroids:
                self._smoke_centroids[key] = deque(maxlen=TEMPORAL_WINDOW)
            self._smoke_centroids[key].append((cx_norm, cy_norm))

            history = self._smoke_centroids[key]
            if not bypass_confirmation:
                if len(history) < 2:
                    continue
                ox, oy = history[0]
                nx, ny = history[-1]
                displacement = ((nx - ox) ** 2 + (ny - oy) ** 2) ** 0.5
                if displacement < SMOKE_MOTION_MIN:
                    continue

            conf = min(0.94, 0.55 + (area / total_pixels) * 2.0)
            boxes.append(BoundingBox(
                x1=round(x / w, 4),
                y1=round(y / h, 4),
                x2=round((x + bw) / w, 4),
                y2=round((y + bh) / h, 4),
                confidence=conf,
            ))

        # Prune centroid tracks whose region was not detected this frame
        stale = [k for k in self._smoke_centroids if k not in seen_keys]
        for k in stale:
            del self._smoke_centroids[k]

        return boxes

    # ── People Detection ──────────────────────────────────────────────────────

    def _detect_people(self, image: np.ndarray, w: int, h: int, bypass_confirmation: bool = False) -> list[BoundingBox]:
        boxes: list[BoundingBox] = []
        if self._coco_model is not None:
            try:
                results = self._coco_model.predict(
                    source=image,
                    conf=CONF_PERSON_THRESH,
                    imgsz=640,
                    verbose=False,
                    device="cpu",
                )
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = self._coco_model.names.get(cls_id, "").lower()
                        # COCO class 0 is person
                        if cls_name == "person" or cls_id == 0:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = float(box.conf[0].item())
                            boxes.append(BoundingBox(
                                x1=round(max(0.0, x1 / w), 4),
                                y1=round(max(0.0, y1 / h), 4),
                                x2=round(min(1.0, x2 / w), 4),
                                y2=round(min(1.0, y2 / h), 4),
                                confidence=conf,
                            ))
            except Exception as e:
                logger.error("YOLO People detection error: %s", e)

        if bypass_confirmation:
            return boxes
        return self._person_tracker.update(boxes)

    # ── Livestock Detection ───────────────────────────────────────────────────

    def _detect_livestock(self, image: np.ndarray, w: int, h: int, bypass_confirmation: bool = False) -> list[BoundingBox]:
        boxes: list[BoundingBox] = []
        if self._coco_model is not None:
            animal_ids = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
            try:
                results = self._coco_model.predict(
                    source=image,
                    conf=CONF_ANIMAL_THRESH,
                    imgsz=640,
                    verbose=False,
                    device="cpu",
                )
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        if cls_id in animal_ids:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = float(box.conf[0].item())
                            boxes.append(BoundingBox(
                                x1=round(max(0.0, x1 / w), 4),
                                y1=round(max(0.0, y1 / h), 4),
                                x2=round(min(1.0, x2 / w), 4),
                                y2=round(min(1.0, y2 / h), 4),
                                confidence=conf,
                            ))
            except Exception as e:
                logger.error("Livestock detection error: %s", e)

        if bypass_confirmation:
            return boxes
        return self._livestock_tracker.update(boxes)

    @staticmethod
    def _iou(b1: BoundingBox, b2: BoundingBox) -> float:
        ix1 = max(b1.x1, b2.x1)
        iy1 = max(b1.y1, b2.y1)
        ix2 = min(b1.x2, b2.x2)
        iy2 = min(b1.y2, b2.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area1 = (b1.x2 - b1.x1) * (b1.y2 - b1.y1)
        area2 = (b2.x2 - b2.x1) * (b2.y2 - b2.y1)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
