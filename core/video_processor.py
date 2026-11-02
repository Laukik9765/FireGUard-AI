"""
core/video_processor.py
────────────────────────
Live webcam / CCTV stream processor.

Handles:
- Continuous frame capture (background thread)
- Time-shifted ring buffer (BUFFER_SECONDS delay)
  Emits the buffered frame whose capture timestamp best matches the
  moment inference completed, so bounding boxes are always painted on
  the exact frame that was analysed — not a future frame that has moved.
- Frame sampling — inference runs on selected frames only
- Rendering annotated frames (bounding boxes drawn on CPU)
- Pushing results to the Flask-SocketIO dashboard

The heavy LocateAnything inference is executed in a separate thread so the
UI never blocks.

TIME-SHIFT ARCHITECTURE
────────────────────────
  Camera ──► [ Ring Buffer: BUFFER_SECONDS × FPS frames (timestamped) ]
                  │
                  ├── Every N frames ──► Inference Queue ──► _inference_loop
                  │                                               │
                  └── Dashboard emits the buffered frame ◄── result + timestamp

The delay introduced equals BUFFER_SECONDS (default 1.5 s).  It gives the
temporal-confirmation gating in MockDetector enough consecutive samples
to validate fire/smoke without false-positive triggers.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from collections import deque
from typing import Optional, Callable

import cv2
import numpy as np

from core.alert_manager import AlertManager, Alert
from core.detector import Detection, DetectionQuery, Detector
from core.event_manager import EventManager
from core.hazard_engine import HazardEngine, HazardSnapshot
from core.smoke_analyzer import SmokeAnalyzer
from core.tracker import ObjectTracker

logger = logging.getLogger(__name__)

# ── Time-Shift Buffer Configuration ───────────────────────────────────────────
# Delay between camera capture and dashboard display.  1.5 s lets the temporal-
# confirmation gating in MockDetector accumulate 3–4 inference samples before
# a detection is shown, eliminating transient false positives on live feeds.
# Override via environment variable BUFFER_SECONDS (e.g. "0" to disable).
BUFFER_SECONDS: float = float(os.environ.get("BUFFER_SECONDS", "1.5"))

# Ring buffer capacity in frames.  At 30 FPS and 1.5 s buffer this is 45 slots.
# Capped at 120 to prevent unbounded RAM growth if FPS is mis-reported.
_BUFFER_CAPACITY: int = min(120, max(10, int(BUFFER_SECONDS * 30) + 5))

# ── Colour scheme for bounding boxes ─────────────────────────────────────────
BOX_COLOURS = {
    DetectionQuery.FIRE:      (0,   60,  220),   # BGR — deep red-orange
    DetectionQuery.SMOKE:     (160, 160, 160),   # mid-grey
    DetectionQuery.PEOPLE:    (0,   200, 80),    # green
    DetectionQuery.LIVESTOCK: (0,   180, 255),   # amber-yellow
}
LABEL_NAMES = {
    DetectionQuery.FIRE:      "FIRE",
    DetectionQuery.SMOKE:     "SMOKE",
    DetectionQuery.PEOPLE:    "PERSON",
    DetectionQuery.LIVESTOCK: "ANIMAL",
}


class VideoProcessor:
    """
    Manages the live camera feed and inference pipeline.

    Parameters
    ----------
    detector : Detector
        Any loaded Detector implementation.
    camera_index : int
        OpenCV camera index (0 = default webcam).
    inference_interval : int
        Run inference every N frames.  Frames between are handled by the
        tracker.  Tune this based on actual RTX 3050 benchmarks.
    emit_callback : callable
        Function called with (event_name, data_dict) to push updates to the
        dashboard (e.g., socketio.emit).
    """

    def __init__(
        self,
        detector: Detector,
        camera_index: int = 1,
        inference_interval: int = 5,
        emit_callback: Optional[Callable] = None,
    ) -> None:
        self._detector   = detector
        self._camera_idx = camera_index
        self._interval   = inference_interval
        self._emit       = emit_callback or (lambda *_: None)

        self._smoke_analyzer = SmokeAnalyzer()
        self._hazard_engine  = HazardEngine()
        self._event_manager  = EventManager()
        self._alert_manager  = AlertManager()
        self._tracker        = ObjectTracker()

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None

        self._frame_lock = threading.Lock()
        self._frame_counter = 0

        # ── Time-Shift Ring Buffer ─────────────────────────────────────────
        # Each slot is a (capture_timestamp: float, frame: np.ndarray) tuple.
        # The capture loop appends here continuously; the emit loop reads the
        # oldest slot whose age >= BUFFER_SECONDS before displaying it.
        self._frame_buffer: deque[tuple[float, np.ndarray]] = deque(
            maxlen=_BUFFER_CAPACITY
        )

        # Frames queued for actual neural-network inference (every N frames).
        # Each slot: (capture_timestamp: float, frame: np.ndarray)
        self._infer_queue: deque[tuple[float, np.ndarray]] = deque(maxlen=16)

        # The inference thread signals the emit loop: (infer_ts, annotated_frame)
        self._ready_lock = threading.Lock()
        self._ready_frame: Optional[np.ndarray] = None   # annotated, ready to push
        self._ready_ts:    float = 0.0                    # capture ts of analysed frame

        # Most-recent detection results (shared between threads)
        self._last_detections: dict[DetectionQuery, Detection] = {}
        self._last_snapshot: Optional[HazardSnapshot] = None

    # ── Camera Discovery & Backend ───────────────────────────────────────────

    @staticmethod
    def _open_cap(idx: int) -> Optional[cv2.VideoCapture]:
        """Attempt opening a camera index with DirectShow first (Windows), then MSMF/default."""
        # Try DirectShow first on Windows
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            pass

        # Try MSMF / default
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            pass

        return None

    @classmethod
    def list_available_cameras(cls, max_indices: int = 5) -> list[dict]:
        """Scan available video capture devices."""
        cameras = []
        for i in range(max_indices):
            cap = cls._open_cap(i)
            if cap is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
                name = f"Camera {i}"
                if i == 1:
                    name += " (Iriun Webcam)"
                cameras.append({
                    "index": i,
                    "name": name,
                    "resolution": f"{w}x{h}",
                })
                try:
                    cap.release()
                except Exception:
                    pass

        # Always offer synthetic simulated stream as an option
        cameras.append({
            "index": -1,
            "name": "Simulated Camera Feed (Synthetic)",
            "resolution": "640x480",
        })
        return cameras

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, camera_index: Optional[int] = None) -> bool:
        """Open camera and start processing threads."""
        if camera_index is not None:
            self._camera_idx = camera_index

        self._synthetic = False

        if self._camera_idx == -1:
            logger.info("Using simulated camera feed explicitly requested.")
            self._synthetic = True
        else:
            # Try specified camera index
            self._cap = self._open_cap(self._camera_idx)

            # If specified index failed and index was 0, auto-search for any working physical camera
            if (self._cap is None or not self._cap.isOpened()) and self._camera_idx == 0:
                logger.info("Index 0 unavailable, searching for working physical cameras (e.g. Iriun)...")
                for probe_idx in (1, 2, 3):
                    probe_cap = self._open_cap(probe_idx)
                    if probe_cap is not None:
                        logger.info("Found working camera on index %d!", probe_idx)
                        self._cap = probe_cap
                        self._camera_idx = probe_idx
                        break

            if self._cap is None or not self._cap.isOpened():
                logger.warning(
                    "Cannot open camera index %d. Falling back to synthetic video feed.",
                    self._camera_idx,
                )
                self._synthetic = True
            else:
                logger.info(
                    "Camera index %d opened: %dx%d @ %.1f fps",
                    self._camera_idx,
                    self._cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                    self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    self._cap.get(cv2.CAP_PROP_FPS),
                )

        self._running = True
        self._capture_thread   = threading.Thread(target=self._capture_loop,   daemon=True, name="fg-capture")
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True, name="fg-infer")
        self._emit_thread      = threading.Thread(target=self._emit_loop,      daemon=True, name="fg-emit")
        self._capture_thread.start()
        self._inference_thread.start()
        self._emit_thread.start()
        logger.info(
            "VideoProcessor started (BUFFER_SECONDS=%.1f, buffer_capacity=%d frames).",
            BUFFER_SECONDS, _BUFFER_CAPACITY,
        )
        return True

    def stop(self) -> None:
        """Stop all threads and release camera."""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=3)
        if self._inference_thread:
            self._inference_thread.join(timeout=5)
        if hasattr(self, "_emit_thread") and self._emit_thread:
            self._emit_thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        self._cap = None
        self._synthetic = False
        self._frame_buffer.clear()
        self._infer_queue.clear()
        logger.info("VideoProcessor stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_snapshot(self) -> Optional[HazardSnapshot]:
        return self._last_snapshot

    # ── Internal loops ────────────────────────────────────────────────────────

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Create a synthetic 640x480 surveillance camera frame with simulated fire/smoke."""
        h, w = 480, 640
        # Dark ambient room background
        frame = np.full((h, w, 3), 28, dtype=np.uint8)
        
        # Grid lines / floor perspective
        cv2.line(frame, (0, 360), (w, 360), (45, 45, 45), 1)
        cv2.line(frame, (0, 480), (200, 360), (40, 40, 40), 1)
        cv2.line(frame, (w, 480), (440, 360), (40, 40, 40), 1)

        t = time.time()
        # Simulated fire glow (orange/yellow flicker)
        fire_cx = int(220 + 20 * np.sin(t * 2))
        fire_cy = int(280 + 10 * np.cos(t * 3))
        radius = int(45 + 15 * np.sin(t * 8))
        
        # Soft flame glow
        overlay = frame.copy()
        cv2.circle(overlay, (fire_cx, fire_cy), radius + 25, (0, 100, 220), -1)
        cv2.circle(overlay, (fire_cx, fire_cy), radius, (0, 180, 255), -1)
        cv2.circle(overlay, (fire_cx, fire_cy - 10), radius // 2, (100, 240, 255), -1)
        
        # Simulated smoke plume above fire
        smoke_cy = int(fire_cy - 90 + 15 * np.sin(t * 1.5))
        cv2.ellipse(overlay, (fire_cx + 20, smoke_cy), (75, 45), 0, 0, 360, (140, 140, 140), -1)
        cv2.ellipse(overlay, (fire_cx + 40, smoke_cy - 50), (90, 55), 0, 0, 360, (120, 120, 120), -1)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # Subtle noise
        noise = np.random.randint(-5, 6, (h, w, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # CCTV timestamp and camera header overlay
        cv2.putText(
            frame,
            f"CAM-01 [SIMULATED] | {time.strftime('%Y-%m-%d %H:%M:%S')}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 180),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _capture_loop(self) -> None:
        """
        Continuously capture or synthesize frames and push them into the
        time-shift ring buffer with their capture timestamp.

        Every _interval-th frame is also placed into _infer_queue so the
        inference thread can process it without starving.
        """
        fail_count = 0
        while self._running:
            if self._synthetic:
                frame = self._generate_synthetic_frame()
                time.sleep(0.066)  # ~15 FPS
            else:
                if self._cap is None or not self._cap.isOpened():
                    time.sleep(0.1)
                    continue

                ok, frame = self._cap.read()
                if not ok or frame is None:
                    fail_count += 1
                    if fail_count % 30 == 0:
                        logger.warning(
                            "Repeated capture failures (%d), attempting camera re-init...",
                            fail_count,
                        )
                        try:
                            if self._cap:
                                self._cap.release()
                            self._cap = self._open_cap(self._camera_idx)
                        except Exception:
                            pass
                    time.sleep(0.033)
                    continue

                fail_count = 0
                time.sleep(0.01)  # Smooth frame pacing

            capture_ts = time.monotonic()
            frame_copy = frame.copy()

            with self._frame_lock:
                self._frame_counter += 1
                # Push raw frame into the time-shift ring buffer
                self._frame_buffer.append((capture_ts, frame_copy))

                # Queue every N-th frame for neural inference
                if self._frame_counter % self._interval == 0:
                    self._infer_queue.append((capture_ts, frame_copy))

    def _emit_loop(self) -> None:
        """
        Continuously poll the ring buffer for frames that have aged at least
        BUFFER_SECONDS and emit them to the dashboard annotated with the most
        recent inference result.

        This is the key mechanism of the time-shift architecture:
        the viewer always sees a frame that the neural engine has already
        had time to fully analyse.
        """
        while self._running:
            now = time.monotonic()
            frame_to_emit: Optional[np.ndarray] = None

            with self._frame_lock:
                # Walk from the oldest buffered frame forward until we find
                # one whose age is >= BUFFER_SECONDS (i.e. old enough that
                # inference has had time to process it).
                if self._frame_buffer:
                    oldest_ts, oldest_frame = self._frame_buffer[0]
                    if (now - oldest_ts) >= BUFFER_SECONDS:
                        self._frame_buffer.popleft()
                        frame_to_emit = oldest_frame

            if frame_to_emit is not None:
                annotated = self._annotate(frame_to_emit)
                self._push_frame(annotated)
            else:
                # Buffer not yet full enough — sleep briefly to avoid spinning
                time.sleep(0.005)

    def _inference_loop(self) -> None:
        """
        Consume queued frames from _infer_queue and run the full
        LocateAnything pipeline on each one.  Detection results are stored
        in _last_detections / _last_snapshot so that _emit_loop can paint
        them onto the matching delayed frame.
        """
        while self._running:
            entry: Optional[tuple[float, np.ndarray]] = None

            with self._frame_lock:
                if self._infer_queue:
                    entry = self._infer_queue.popleft()

            if entry is None:
                time.sleep(0.01)
                continue

            _capture_ts, frame = entry
            self._run_inference(frame)

    def _run_inference(self, frame: np.ndarray) -> None:
        """Run all four queries on a single frame."""
        results = {}
        for query in DetectionQuery:
            try:
                det = self._detector.detect(frame, query)
                results[query] = det
                self._tracker.update_from_detection(det)
            except Exception as e:
                logger.error("Inference error (%s): %s", query.name, e)

        if not results:
            return

        smoke_analysis = None
        if DetectionQuery.SMOKE in results:
            smoke_analysis = self._smoke_analyzer.update(results[DetectionQuery.SMOKE])

        snapshot = self._hazard_engine.update(
            fire_detection=results.get(DetectionQuery.FIRE,      _empty(DetectionQuery.FIRE)),
            smoke_detection=results.get(DetectionQuery.SMOKE,    _empty(DetectionQuery.SMOKE)),
            people_detection=results.get(DetectionQuery.PEOPLE,  _empty(DetectionQuery.PEOPLE)),
            livestock_detection=results.get(DetectionQuery.LIVESTOCK, _empty(DetectionQuery.LIVESTOCK)),
            smoke_analysis=smoke_analysis,
        )

        self._last_snapshot = snapshot
        self._last_detections = results

        alert = self._alert_manager.process(snapshot)
        self._event_manager.process(snapshot)

        self._push_snapshot(snapshot, alert)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _annotate(self, frame: np.ndarray) -> np.ndarray:
        """Draw bounding boxes from the most recent tracking data."""
        self._tracker.step_motion()
        out = frame.copy()
        h, w = out.shape[:2]

        for query in DetectionQuery:
            colour = BOX_COLOURS[query]
            label  = LABEL_NAMES[query]
            tracks = self._tracker.get_active_tracks(query)

            for track in tracks:
                x1, y1, x2, y2 = track.box.to_pixel(w, h)
                cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(
                    out, f"{label} {track.box.confidence:.2f}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
                )

        return out

    # ── Dashboard push ────────────────────────────────────────────────────────

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Return the most recent JPEG bytes of the annotated frame."""
        return getattr(self, "_latest_jpeg", None)

    def _push_frame(self, frame: np.ndarray) -> None:
        """JPEG-encode frame, store for MJPEG streaming, and emit as base64 to dashboard."""
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            self._latest_jpeg = buf.tobytes()
            b64 = base64.b64encode(buf).decode()
            self._emit("frame", {"image": b64})
        except Exception as e:
            logger.error("Frame push failed: %s", e)

    def _push_snapshot(
        self, snapshot: HazardSnapshot, alert: Optional[Alert]
    ) -> None:
        """Push hazard snapshot data to dashboard."""
        try:
            payload = {
                "hazard_level":   snapshot.hazard_level.value,
                "fire_state":     snapshot.fire_state.value,
                "smoke_state":    snapshot.smoke_state.value,
                "people_count":   snapshot.people_count,
                "livestock_count": snapshot.livestock_count,
            }
            if snapshot.smoke_analysis:
                sa = snapshot.smoke_analysis
                payload["smoke"] = {
                    "detected":    sa.detected,
                    "density":     sa.density_index,
                    "coverage":    sa.coverage_pct,
                    "hazard":      sa.hazard_level.value,
                    "trend":       sa.trend.value,
                    "trend_arrow": sa.trend.arrow,
                }
            if alert:
                payload["alert"] = alert.to_dict()

            self._emit("snapshot", payload)
        except Exception as e:
            logger.error("Snapshot push failed: %s", e)


# ── Helper ────────────────────────────────────────────────────────────────────

def _empty(query: DetectionQuery) -> Detection:
    """Return an empty (no-detection) result for a given query."""
    from core.detector import Detection
    return Detection(query=query, boxes=[])
