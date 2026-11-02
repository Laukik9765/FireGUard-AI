"""
verify_detection_pipeline.py
────────────────────────────
Runs detection verification on realistic test imagery for:
  1. Fire / Flames scene
  2. Smoke plume scene
  3. People / Human scene

Annotates results with color-coded bounding boxes and saves outputs to `outputs/`.
"""

import json
from pathlib import Path
import cv2
import numpy as np

from core.detector import DetectionQuery
from models.locate_anything.mock_detector import MockDetector

PROJECT_ROOT = Path(__file__).resolve().parent
INPUTS_DIR = PROJECT_ROOT / "inputs" / "images"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

COLOR_MAP = {
    DetectionQuery.FIRE: (0, 69, 255),       # Vibrant Red/Orange
    DetectionQuery.SMOKE: (128, 128, 128),   # Grey
    DetectionQuery.PEOPLE: (255, 165, 0),    # Cyan/Blue
    DetectionQuery.LIVESTOCK: (0, 255, 0),   # Green
}


def run_verification():
    print("=" * 70)
    print(" FireGuard AI — System Detection Verification")
    print("=" * 70)

    detector = MockDetector()
    detector.load()

    test_cases = [
        {
            "name": "Fire / Flame Scene",
            "file": INPUTS_DIR / "test_fire_real.jpg",
            "primary_query": DetectionQuery.FIRE,
            "out_file": OUTPUTS_DIR / "verified_fire_annotated.jpg",
        },
        {
            "name": "Smoke & Fire Emergency Scene",
            "file": INPUTS_DIR / "test_smoke_real.jpg",
            "primary_query": DetectionQuery.SMOKE,
            "out_file": OUTPUTS_DIR / "verified_smoke_annotated.jpg",
        },
        {
            "name": "People Scene (Anti-False-Positive Test: orange jacket/tent)",
            "file": INPUTS_DIR / "test_people_real.jpg",
            "primary_query": DetectionQuery.PEOPLE,
            "out_file": OUTPUTS_DIR / "verified_people_annotated.jpg",
        },
    ]

    report = []

    for tc in test_cases:
        img_path = tc["file"]
        if not img_path.exists():
            print(f"[-] Missing test file: {img_path}")
            continue

        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]
        annotated = frame.copy()

        print(f"\n[+] Testing: {tc['name']} ({img_path.name} — {w}x{h})")

        case_results = {"scene": tc["name"], "file": img_path.name, "detections": {}}

        for q in [DetectionQuery.FIRE, DetectionQuery.SMOKE, DetectionQuery.PEOPLE]:
            # Run query (calling 3 times to satisfy temporal persistence if needed for static verification)
            det = None
            for _ in range(3):
                det = detector.detect(frame, q)

            boxes = det.boxes if det else []
            case_results["detections"][q.name] = len(boxes)

            color = COLOR_MAP[q]
            for box in boxes:
                x1, y1 = int(box.x1 * w), int(box.y1 * h)
                x2, y2 = int(box.x2 * w), int(box.y2 * h)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                label_txt = f"{q.name} ({box.confidence:.2f})"
                cv2.putText(
                    annotated,
                    label_txt,
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    label_txt,
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                print(f"    - Found {q.name} box: [{box.x1:.3f}, {box.y1:.3f}, {box.x2:.3f}, {box.y2:.3f}] (conf={box.confidence:.2f})")

        cv2.imwrite(str(tc["out_file"]), annotated)
        print(f"    Saved annotated visualization to: {tc['out_file']}")
        report.append(case_results)

    print("\n" + "=" * 70)
    print(" Verification Summary:")
    print("=" * 70)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_verification()
