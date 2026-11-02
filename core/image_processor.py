"""
core/image_processor.py
────────────────────────
Single-image and video-file analysis.

Handles:
- Image upload → analysis → annotated output + JSON report
- Video upload → frame-by-frame analysis → timeline + annotated video
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
from PIL import Image

from core.alert_manager import AlertManager
from core.detector import Detection, DetectionQuery, Detector
from core.event_manager import EventManager
from core.hazard_engine import HazardEngine, HazardSnapshot
from core.smoke_analyzer import SmokeAnalyzer
from core.tracker import ObjectTracker
from core.video_processor import _empty, BOX_COLOURS, LABEL_NAMES

logger = logging.getLogger(__name__)

OUTPUT_ANNOTATED = Path("outputs/annotated")
OUTPUT_REPORTS   = Path("outputs/reports")

HAZARD_RANKS = {
    "NORMAL": 0,
    "LOW": 1,
    "MODERATE": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


def _extract_boxes_info(
    detections: dict[DetectionQuery, Detection],
) -> list[dict]:
    """Extract serialisable bounding box metadata for UI overlay rendering."""
    boxes_info = []
    for query, det in detections.items():
        label_prefix = LABEL_NAMES.get(query, query.name)
        q_key = query.name.lower()
        for box in det.boxes:
            boxes_info.append({
                "query": q_key,
                "label": f"{label_prefix} {box.confidence:.2f}",
                "confidence": round(float(box.confidence), 2),
                "top": f"{box.y1 * 100:.1f}%",
                "left": f"{box.x1 * 100:.1f}%",
                "width": f"{max(0.0, box.x2 - box.x1) * 100:.1f}%",
                "height": f"{max(0.0, box.y2 - box.y1) * 100:.1f}%",
            })
    return boxes_info


class ImageProcessor:
    """Analyses a single uploaded image."""

    def __init__(self, detector: Detector) -> None:
        self._detector = detector

    def analyse(
        self,
        image_path: Path,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        """
        Run all queries on the image.

        Parameters
        ----------
        image_path : Path
            Path to the uploaded image file (never overwritten).
        progress_cb : callable, optional
            Called with (step: str) for UI progress updates.

        Returns
        -------
        dict
            JSON-serialisable analysis result including detections and paths
            to the annotated output.
        """
        t0 = time.time()
        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")

        h, w = frame.shape[:2]
        results: dict[DetectionQuery, Detection] = {}

        for query in DetectionQuery:
            if progress_cb:
                progress_cb(f"Detecting {query.name.lower()} …")
            results[query] = self._detector.detect(frame, query)

        smoke_analyzer = SmokeAnalyzer()
        smoke_analysis = smoke_analyzer.update(results[DetectionQuery.SMOKE])

        hazard_engine = HazardEngine()
        snapshot = hazard_engine.update(
            fire_detection=results[DetectionQuery.FIRE],
            smoke_detection=results[DetectionQuery.SMOKE],
            people_detection=results[DetectionQuery.PEOPLE],
            livestock_detection=results[DetectionQuery.LIVESTOCK],
            smoke_analysis=smoke_analysis,
        )

        annotated = _draw_boxes(frame, results)
        out_path = _save_annotated_image(annotated, image_path)

        smoke_score = round(float(smoke_analysis.density_index) / 100.0, 2)
        proc_time = round(time.time() - t0, 2)
        boxes_info = _extract_boxes_info(results)

        report = {
            "source":             str(image_path.name),
            "annotated":          str(out_path),
            "annotated_url":      f"/outputs/annotated/{out_path.name}",
            "preview_url":        f"/outputs/annotated/{out_path.name}",
            "duration_sec":       0,
            "duration_formatted": "00:00:01",
            "processing_time":    proc_time,
            "fire": {
                "detected": results[DetectionQuery.FIRE].detected,
                "count":    results[DetectionQuery.FIRE].count,
            },
            "smoke": {
                "detected":    smoke_analysis.detected,
                "density":     round(float(smoke_analysis.density_index), 1),
                "coverage":    round(float(smoke_analysis.coverage_pct), 1),
                "hazard":      smoke_analysis.hazard_level.value,
                "smoke_score": smoke_score,
                "trend":       smoke_analysis.trend.value,
            },
            "people":       {"count": results[DetectionQuery.PEOPLE].count},
            "livestock":    {"count": results[DetectionQuery.LIVESTOCK].count},
            "hazard_level": snapshot.hazard_level.value,
            "boxes":        boxes_info,
            "timeline":     [{
                "time":          "00:00",
                "timestamp_sec": 0.0,
                "hazard":        snapshot.hazard_level.value,
                "fire":          snapshot.fire_state.value,
                "smoke":         snapshot.smoke_state.value,
                "people":        snapshot.people_count,
                "livestock":     snapshot.livestock_count,
                "smoke_density": round(float(smoke_analysis.density_index), 1),
                "smoke_coverage": round(float(smoke_analysis.coverage_pct), 1),
                "smoke_trend":   smoke_analysis.trend.value,
            }],
        }

        report_path = OUTPUT_REPORTS / f"{image_path.stem}_report.json"
        OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))

        return report


class VideoFileProcessor:
    """Analyses an uploaded video file and produces a detection timeline."""

    def __init__(
        self,
        detector: Detector,
        frame_interval: int = 15,    # Analyse every N frames
        progress_cb: Optional[Callable] = None,
    ) -> None:
        self._detector      = detector
        self._frame_interval = frame_interval
        self._progress_cb   = progress_cb or (lambda *_: None)

    def analyse(self, video_path: Path) -> dict:
        """
        Analyse a video file frame by frame.

        Original file is NEVER modified.

        Returns
        -------
        dict
            timeline, event summary, and path to annotated output video.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps       = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = _annotated_video_path(video_path)
        OUTPUT_ANNOTATED.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        t0 = time.time()
        smoke_analyzer = SmokeAnalyzer()
        hazard_engine  = HazardEngine()
        alert_manager  = AlertManager()
        event_manager  = EventManager()
        tracker        = ObjectTracker()

        timeline: list[dict] = []
        frame_idx = 0
        last_results: dict[DetectionQuery, Detection] = {}
        peak_hazard_score = -1.0
        peak_preview_frame = None
        peak_results: dict[DetectionQuery, Detection] = {}

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_sec = frame_idx / fps

            if frame_idx % self._frame_interval == 0:
                results: dict[DetectionQuery, Detection] = {}
                for query in DetectionQuery:
                    det = self._detector.detect(frame, query, mode="single")
                    results[query] = det
                    tracker.update_from_detection(det)
                last_results = results

                smoke_analysis = smoke_analyzer.update(results[DetectionQuery.SMOKE])
                snapshot = hazard_engine.update(
                    fire_detection=results[DetectionQuery.FIRE],
                    smoke_detection=results[DetectionQuery.SMOKE],
                    people_detection=results[DetectionQuery.PEOPLE],
                    livestock_detection=results[DetectionQuery.LIVESTOCK],
                    smoke_analysis=smoke_analysis,
                )
                alert_manager.process(snapshot)
                event_manager.process(snapshot)

                timeline.append(_timeline_entry(timestamp_sec, snapshot, smoke_analysis))

                # Track peak hazard moment for preview frame
                h_rank = HAZARD_RANKS.get(snapshot.hazard_level.value, 0) * 100
                density_val = float(smoke_analysis.density_index) if smoke_analysis else 0.0
                fire_score = results[DetectionQuery.FIRE].count * 50
                people_score = results[DetectionQuery.PEOPLE].count * 10
                current_score = h_rank + density_val + fire_score + people_score

                annotated = _draw_boxes(frame, last_results)
                if current_score >= peak_hazard_score or peak_preview_frame is None:
                    peak_hazard_score = current_score
                    peak_preview_frame = annotated.copy()
                    peak_results = results

                self._progress_cb({
                    "frame": frame_idx,
                    "total": total_frames,
                    "timestamp": f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}",
                    "hazard": snapshot.hazard_level.value,
                })
            else:
                tracker.age_tracks()
                annotated = _draw_boxes(frame, last_results)

            writer.write(annotated)
            frame_idx += 1

        cap.release()
        writer.release()

        # Save peak frame as preview image
        preview_path = OUTPUT_ANNOTATED / f"{video_path.stem}_preview.jpg"
        if peak_preview_frame is not None:
            cv2.imwrite(str(preview_path), peak_preview_frame)
            preview_url = f"/outputs/annotated/{preview_path.name}"
        else:
            preview_url = f"/outputs/annotated/{out_path.name}"

        # Aggregate summary metrics
        max_smoke_density = max([e.get("smoke_density", 0.0) for e in timeline] or [0.0])
        max_smoke_cov     = max([e.get("smoke_coverage", 0.0) for e in timeline] or [0.0])
        any_fire          = any(e.get("fire", "Normal").lower() != "normal" for e in timeline if e.get("fire"))
        max_people        = max([e.get("people", 0) for e in timeline] or [0])
        max_livestock     = max([e.get("livestock", 0) for e in timeline] or [0])
        latest_trend      = timeline[-1].get("smoke_trend", "Stable") if timeline else "Stable"

        if max_smoke_density >= 75:
            smoke_hazard = "Heavy"
        elif max_smoke_density >= 45:
            smoke_hazard = "Moderate"
        elif max_smoke_density >= 20:
            smoke_hazard = "Low"
        else:
            smoke_hazard = "None"

        peak_hazard = max([e.get("hazard", "NORMAL") for e in timeline] or ["NORMAL"], key=lambda x: HAZARD_RANKS.get(x, 0))
        total_sec = frame_idx / fps if fps > 0 else 0.0
        mins = int(total_sec // 60)
        secs = int(total_sec % 60)
        dur_str = f"{mins:02d}:{secs:02d}"
        proc_time = round(time.time() - t0, 1)

        report = {
            "source":             str(video_path.name),
            "annotated":          str(out_path),
            "annotated_url":      f"/outputs/annotated/{out_path.name}",
            "preview_url":        preview_url,
            "duration_sec":       round(total_sec, 1),
            "duration_formatted": dur_str,
            "processing_time":    proc_time,
            "frames_analysed":    len(timeline),
            "timeline":           timeline,
            "events":             event_manager.list_events(),
            "hazard_level":       peak_hazard,
            "fire": {
                "detected": any_fire,
                "count":    1 if any_fire else 0,
            },
            "smoke": {
                "detected":    max_smoke_density > 0,
                "density":     round(max_smoke_density, 1),
                "coverage":    round(max_smoke_cov, 1),
                "hazard":      smoke_hazard,
                "smoke_score": round(max_smoke_density / 100.0, 2),
                "trend":       latest_trend,
            },
            "people":       {"count": max_people},
            "livestock":    {"count": max_livestock},
            "boxes":        _extract_boxes_info(peak_results),
        }

        rp = OUTPUT_REPORTS / f"{video_path.stem}_report.json"
        OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2))
        return report


# ── Shared helpers ────────────────────────────────────────────────────────────

def _draw_boxes(
    frame: np.ndarray,
    detections: dict[DetectionQuery, Detection],
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    for query, det in detections.items():
        colour = BOX_COLOURS[query]
        label  = LABEL_NAMES[query]
        for box in det.boxes:
            x1, y1, x2, y2 = box.to_pixel(w, h)
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                out, f"{label} {box.confidence:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
            )
    return out


def _save_annotated_image(frame: np.ndarray, source: Path) -> Path:
    OUTPUT_ANNOTATED.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ANNOTATED / f"{source.stem}_annotated{source.suffix}"
    cv2.imwrite(str(out), frame)
    return out


def _annotated_video_path(source: Path) -> Path:
    return OUTPUT_ANNOTATED / f"{source.stem}_annotated.mp4"


def _timeline_entry(
    ts: float,
    snapshot: HazardSnapshot,
    smoke_analysis,
) -> dict:
    mins = int(ts // 60)
    secs = int(ts % 60)
    density = float(smoke_analysis.density_index) if smoke_analysis and smoke_analysis.density_index is not None else 0.0
    coverage = float(smoke_analysis.coverage_pct) if smoke_analysis and smoke_analysis.coverage_pct is not None else 0.0
    trend = smoke_analysis.trend.value if smoke_analysis and smoke_analysis.trend else "Stable"

    entry: dict = {
        "time":          f"{mins:02d}:{secs:02d}",
        "timestamp_sec": round(ts, 2),
        "hazard":        snapshot.hazard_level.value,
        "fire":          snapshot.fire_state.value,
        "smoke":         snapshot.smoke_state.value,
        "people":        snapshot.people_count,
        "livestock":     snapshot.livestock_count,
        "smoke_density": round(density, 1),
        "smoke_coverage": round(coverage, 1),
        "smoke_trend":   trend,
    }
    return entry
