"""
app.py
───────
FireGuard AI — Main Application Entry Point

Starts the Flask-SocketIO server that powers the live dashboard.
All inference happens in background threads — the UI never blocks.

Usage
-----
    python app.py                   # Real model (requires GPU + weights)
    python app.py --mock            # Mock detector (for UI dev / testing)
    python app.py --mock --port 8080
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_socketio import SocketIO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("fireguard")

# ── App factory ───────────────────────────────────────────────────────────────

def create_app(use_mock: bool = False) -> tuple[Flask, SocketIO]:
    app = Flask(
        __name__,
        template_folder="dashboard/templates",
        static_folder="dashboard/static",
    )
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "fireguard-dev-key")
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024   # 512 MB upload limit
    app.config["UPLOAD_FOLDER_IMAGES"] = Path("inputs/images")
    app.config["UPLOAD_FOLDER_VIDEOS"] = Path("inputs/videos")

    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    # ── Detector selection ────────────────────────────────────────────────────
    if use_mock:
        from models.locate_anything.mock_detector import MockDetector
        detector = MockDetector(seed=42)
        logger.info("Using MockDetector (no GPU required).")
    else:
        from models.locate_anything.locate_anything_detector import LocateAnythingDetector
        detector = LocateAnythingDetector()
        logger.info("Using LocateAnythingDetector (GPU required).")

    detector.load()

    # ── Video processor (live monitoring) ────────────────────────────────────
    from core.video_processor import VideoProcessor
    video_proc = VideoProcessor(
        detector=detector,
        camera_index=1,
        emit_callback=lambda event, data: socketio.emit(event, data),
    )

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/graph")
    def show_graph():
        from flask import send_file
        graph_file = Path("graphify-out/graph.html")
        if graph_file.exists():
            return send_file(graph_file)
        return "Graph not generated yet.", 404

    @app.route("/video_feed")
    def video_feed():
        import time
        def generate():
            while True:
                jpeg = video_proc.get_latest_jpeg()
                if jpeg:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(0.04)
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/status")
    def api_status():
        return jsonify({
            "status": "online",
            "camera_running": video_proc.is_running,
            "detector_loaded": detector.is_loaded(),
            "vram_mb": detector.vram_usage_mb,
        })

    @app.route("/api/cameras")
    def list_cameras():
        cameras = VideoProcessor.list_available_cameras()
        return jsonify(cameras)

    @app.route("/api/camera/start", methods=["POST"])
    def camera_start():
        data = request.get_json(silent=True) or {}
        idx  = data.get("camera_index", None)
        if idx is not None:
            try:
                idx = int(idx)
            except (ValueError, TypeError):
                idx = None
        if video_proc.is_running:
            return jsonify({"ok": True, "already_running": True})
        ok = video_proc.start(camera_index=idx)
        return jsonify({"ok": ok, "error": None if ok else "Failed to open camera."})

    @app.route("/api/camera/stop", methods=["POST"])
    def camera_stop():
        video_proc.stop()
        return jsonify({"ok": True})

    @app.route("/api/analyse/image", methods=["POST"])
    def analyse_image():
        from core.image_processor import ImageProcessor
        from werkzeug.utils import secure_filename

        if "file" not in request.files:
            return jsonify({"ok": False, "error": "No file provided."}), 400

        f = request.files["file"]
        if f.filename == "":
            return jsonify({"ok": False, "error": "Empty filename."}), 400

        filename = secure_filename(f.filename)
        save_dir = app.config["UPLOAD_FOLDER_IMAGES"]
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        f.save(save_path)

        try:
            processor = ImageProcessor(detector)
            report = processor.analyse(save_path)
            return jsonify({"ok": True, "report": report})
        except Exception as e:
            logger.error("Image analysis failed: %s", e)
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/analyse/video", methods=["POST"])
    def analyse_video():
        from core.image_processor import VideoFileProcessor
        from werkzeug.utils import secure_filename

        if "file" not in request.files:
            return jsonify({"ok": False, "error": "No file provided."}), 400

        f = request.files["file"]
        if f.filename == "":
            return jsonify({"ok": False, "error": "Empty filename."}), 400

        filename = secure_filename(f.filename)
        save_dir = app.config["UPLOAD_FOLDER_VIDEOS"]
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        f.save(save_path)

        def progress(data):
            socketio.emit("video_progress", data)

        try:
            processor = VideoFileProcessor(detector, progress_cb=progress)
            report = processor.analyse(save_path)
            return jsonify({"ok": True, "report": report})
        except Exception as e:
            logger.error("Video analysis failed: %s", e)
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/events")
    def api_events():
        from core.event_manager import EventManager
        em = EventManager()
        return jsonify(em.list_events())

    @app.route("/outputs/annotated/<path:filename>")
    def serve_annotated(filename):
        annotated_dir = Path("outputs/annotated").resolve()
        annotated_dir.mkdir(parents=True, exist_ok=True)
        return send_from_directory(str(annotated_dir), filename)

    @app.route("/outputs/reports/<path:filename>")
    def serve_report(filename):
        reports_dir = Path("outputs/reports").resolve()
        reports_dir.mkdir(parents=True, exist_ok=True)
        return send_from_directory(str(reports_dir), filename)

    @app.route("/api/report/pdf", methods=["POST"])
    def generate_pdf():
        """
        Accept the JSON analysis report payload and return a rendered PDF.

        Request body (JSON):  { "report": { ...analysis report dict... } }
        Response:             application/pdf  (binary)
        """
        from core.report_generator import generate_pdf_report
        from flask import Response as FlaskResponse

        data = request.get_json(silent=True) or {}
        report = data.get("report")
        if not report:
            return jsonify({"ok": False, "error": "No report data provided."}), 400
        try:
            pdf_bytes = generate_pdf_report(report)
            source    = str(report.get("source", "analysis")).replace(" ", "_")
            filename  = f"{source}_fireguard_report.pdf"
            return FlaskResponse(
                pdf_bytes,
                mimetype="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename=\"{filename}\"",
                    "Content-Length": str(len(pdf_bytes)),
                },
            )
        except Exception as e:
            logger.error("PDF generation failed: %s", e, exc_info=True)
            return jsonify({"ok": False, "error": str(e)}), 500

    # ── SocketIO events ───────────────────────────────────────────────────────

    @socketio.on("connect")
    def on_connect():
        logger.info("Dashboard client connected: %s", request.sid)

    @socketio.on("disconnect")
    def on_disconnect():
        logger.info("Dashboard client disconnected: %s", request.sid)

    return app, socketio


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FireGuard AI Dashboard")
    parser.add_argument("--mock",   action="store_true", help="Use mock detector (no GPU)")
    parser.add_argument("--host",   default="127.0.0.1",  help="Host to bind")
    parser.add_argument("--port",   default=5000, type=int, help="Port to listen on")
    parser.add_argument("--debug",  action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    logger.info("Starting FireGuard AI on http://%s:%d", args.host, args.port)
    logger.info("Mock mode: %s", args.mock)

    app, socketio = create_app(use_mock=args.mock)
    socketio.run(app, host=args.host, port=args.port, debug=args.debug, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
