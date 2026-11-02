"""
core/tracker.py
────────────────
Lightweight object tracker.

Between expensive LocateAnything inference calls, this tracker keeps bounding
boxes alive by propagating their last known positions.  It does NOT require a
GPU — it runs on the CPU between inference frames.

Current implementation: simple identity-based persistence.
Future: can be upgraded to SORT, ByteTrack, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.detector import BoundingBox, Detection, DetectionQuery


@dataclass
class TrackedObject:
    """A single tracked detection that persists and glides smoothly across frames."""
    query:       DetectionQuery
    box:         BoundingBox
    track_id:    int
    age:         int = 0           # frames since last inference update
    max_age:     int = 20          # frames before track is dropped
    vx:          float = 0.0       # horizontal velocity per frame (normalized coords)
    vy:          float = 0.0       # vertical velocity per frame (normalized coords)


class ObjectTracker:
    """
    Propagates and smoothly interpolates object bounding boxes across frames
    between heavy neural-network inference calls.
    """

    def __init__(self, max_age: int = 20) -> None:
        self._max_age = max_age
        self._tracks: list[TrackedObject] = []
        self._next_id = 0

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

    def update_from_detection(self, detection: Detection) -> None:
        """
        Associate new detection bounding boxes with existing tracks using IoU,
        estimate target velocity vectors, apply Non-Maximum Suppression (NMS),
        and prune stale unmatched tracks rapidly.
        """
        # 1. Apply NMS to incoming detection boxes to prevent overlapping box chaos
        raw_boxes = sorted(detection.boxes, key=lambda b: b.confidence, reverse=True)
        nms_boxes: list[BoundingBox] = []
        for box in raw_boxes:
            if not any(self._iou(box, existing) >= 0.35 for existing in nms_boxes):
                nms_boxes.append(box)

        query_tracks = [t for t in self._tracks if t.query == detection.query]
        other_tracks = [t for t in self._tracks if t.query != detection.query]

        matched_track_ids = set()
        updated_tracks: list[TrackedObject] = []

        for new_box in nms_boxes:
            best_iou = 0.0
            best_track: Optional[TrackedObject] = None

            for t in query_tracks:
                if t.track_id in matched_track_ids:
                    continue
                iou = self._iou(t.box, new_box)
                if iou > best_iou:
                    best_iou = iou
                    best_track = t

            if best_track is not None and best_iou >= 0.12:
                matched_track_ids.add(best_track.track_id)

                # Estimate centroid velocity over the frames elapsed since last hit
                dt_frames = max(1, best_track.age + 1)
                cx_old = (best_track.box.x1 + best_track.box.x2) / 2.0
                cy_old = (best_track.box.y1 + best_track.box.y2) / 2.0
                cx_new = (new_box.x1 + new_box.x2) / 2.0
                cy_new = (new_box.y1 + new_box.y2) / 2.0

                calc_vx = (cx_new - cx_old) / dt_frames
                calc_vy = (cy_new - cy_old) / dt_frames

                # Smooth velocity (low-pass filter)
                best_track.vx = 0.5 * best_track.vx + 0.5 * calc_vx
                best_track.vy = 0.5 * best_track.vy + 0.5 * calc_vy

                # Exponential position smoothing to eliminate coordinate snapping
                alpha = 0.65
                nx1 = round(max(0.0, min(1.0, alpha * new_box.x1 + (1.0 - alpha) * best_track.box.x1)), 4)
                ny1 = round(max(0.0, min(1.0, alpha * new_box.y1 + (1.0 - alpha) * best_track.box.y1)), 4)
                nx2 = round(max(0.0, min(1.0, alpha * new_box.x2 + (1.0 - alpha) * best_track.box.x2)), 4)
                ny2 = round(max(0.0, min(1.0, alpha * new_box.y2 + (1.0 - alpha) * best_track.box.y2)), 4)

                best_track.box = BoundingBox(
                    x1=nx1, y1=ny1, x2=nx2, y2=ny2, confidence=new_box.confidence
                )
                best_track.age = 0
                updated_tracks.append(best_track)
            else:
                # Register new track
                updated_tracks.append(
                    TrackedObject(
                        query=detection.query,
                        box=new_box,
                        track_id=self._next_id,
                        age=0,
                        max_age=2,  # Drop unmatched tracks within 2 cycles (~0.15s) to eliminate ghost boxes
                        vx=0.0,
                        vy=0.0,
                    )
                )
                self._next_id += 1

        # Retain unmatched tracks that have not exceeded their short max_age (2 frames)
        for t in query_tracks:
            if t.track_id not in matched_track_ids:
                t.age += 1
                if t.age < 2:
                    updated_tracks.append(t)

        self._tracks = other_tracks + updated_tracks

    def step_motion(self) -> None:
        """
        Propagate all active bounding boxes forward by their velocity vectors.
        Call on every frame render to guarantee 30+ FPS smooth tracking motion.
        """
        for t in self._tracks:
            if abs(t.vx) > 0.0001 or abs(t.vy) > 0.0001:
                w = t.box.x2 - t.box.x1
                h = t.box.y2 - t.box.y1
                nx1 = max(0.0, min(1.0 - w, t.box.x1 + t.vx))
                ny1 = max(0.0, min(1.0 - h, t.box.y1 + t.vy))
                nx2 = max(w, min(1.0, t.box.x2 + t.vx))
                ny2 = max(h, min(1.0, t.box.y2 + t.vy))
                t.box = BoundingBox(
                    x1=round(nx1, 4),
                    y1=round(ny1, 4),
                    x2=round(nx2, 4),
                    y2=round(ny2, 4),
                    confidence=t.box.confidence,
                )
                # Dampen velocity slightly across non-inference frames
                t.vx *= 0.94
                t.vy *= 0.94

    def age_tracks(self) -> None:
        """Increment age of all tracks and remove expired ones."""
        self.step_motion()
        self._tracks = [t for t in self._tracks if t.age < t.max_age]

    def get_active_tracks(self, query: DetectionQuery) -> list[TrackedObject]:
        """Return all non-expired tracks for the given query."""
        return [t for t in self._tracks if t.query == query and t.age < t.max_age]

    def clear(self) -> None:
        self._tracks.clear()
