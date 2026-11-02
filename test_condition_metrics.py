"""
test_condition_metrics.py
"""

import cv2
import numpy as np
from pathlib import Path
from core.detector import DetectionQuery
from models.locate_anything.mock_detector import MockDetector

PROJECT_ROOT = Path(__file__).resolve().parent
INPUTS_DIR = PROJECT_ROOT / "inputs" / "images"

detector = MockDetector()
detector.load()

w, h = 640, 480
fire_img = cv2.resize(cv2.imread(str(INPUTS_DIR / "test_fire_real.jpg")), (w, h))
smoke_img = cv2.resize(cv2.imread(str(INPUTS_DIR / "test_smoke_real.jpg")), (w, h))
people_img = cv2.resize(cv2.imread(str(INPUTS_DIR / "test_people_real.jpg")), (w, h))

# Test conditions
conditions = {
    "Clear Fire": (DetectionQuery.FIRE, True),
    "Light Smoke": (DetectionQuery.SMOKE, True),
    "Dense Smoke": (DetectionQuery.SMOKE, True),
    "Low Illumination": (DetectionQuery.FIRE, True),
    "Small/Distant Target": (DetectionQuery.FIRE, True),
    "Partial Occlusion": (DetectionQuery.FIRE, True),
    "Smoke-Like Non-Fire Scene": (DetectionQuery.SMOKE, False),
}

np.random.seed(42)
N = 30
summary = {}

for name, (query, has_gt) in conditions.items():
    tp, fp, fn, tn = 0, 0, 0, 0
    detector.unload()
    detector.load()
    
    for i in range(N):
        if name == "Clear Fire":
            frame = fire_img.copy()
            noise = np.random.randint(-6, 7, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
        elif name == "Light Smoke":
            # Real smoke image blended with slight background to represent thin smoke
            bg = cv2.GaussianBlur(people_img, (35, 35), 0)
            frame = cv2.addWeighted(smoke_img, 0.45, bg, 0.55, 0)
            
        elif name == "Dense Smoke":
            # Heavy smoke plume from real image cropped/focused
            frame = smoke_img.copy()
            # Crop to plume region and resize
            plume = smoke_img[50:400, 150:550]
            frame = cv2.resize(plume, (w, h))
            
        elif name == "Low Illumination":
            # Dark background ambient (night scene), fire core preserved as radiant
            frame = (fire_img.astype(np.float32) * 0.20).astype(np.uint8)
            # Find fire mask and restore its radiant combustion luminance
            hsv = cv2.cvtColor(fire_img, cv2.COLOR_BGR2HSV)
            flame_mask = cv2.inRange(hsv, np.array([0, 120, 160]), np.array([30, 255, 255]))
            frame[flame_mask > 0] = fire_img[flame_mask > 0]
            
        elif name == "Small/Distant Target":
            # Distant fire target: small flame patch (35x45 px)
            frame = np.full((h, w, 3), [40, 50, 45], dtype=np.uint8)
            patch = cv2.resize(fire_img[200:350, 250:400], (35, 45))
            py, px = 200 + (i % 5) * 10, 280 + (i % 7) * 10
            frame[py:py+45, px:px+35] = patch
            
        elif name == "Partial Occlusion":
            # Fire with 30-40% slats occlusion
            frame = fire_img.copy()
            for bx in range(160, 480, 45):
                cv2.rectangle(frame, (bx, 0), (bx + 16, h), (25, 25, 25), -1)
                
        elif name == "Smoke-Like Non-Fire Scene":
            # Fog, steam, light grey walls, clouds, dust - NO combustion smoke
            frame = np.full((h, w, 3), [200, 205, 210], dtype=np.uint8)
            # Add textured grid / background walls
            for gx in range(0, w, 40):
                cv2.line(frame, (gx, 0), (gx, h), (170, 175, 180), 2)
            # Add steam/fog cloud overlay (smooth gradient, high frequency edge suppression)
            cv2.ellipse(frame, (320, 240), (180, 120), 0, 0, 360, (215, 218, 222), -1)

        det = None
        for _ in range(2):
            det = detector.detect(frame, query)
            
        is_det = len(det.boxes) > 0 if det else False
        
        if has_gt:
            if is_det:
                tp += 1
            else:
                fn += 1
        else:
            if is_det:
                fp += 1
            else:
                tn += 1
                
    p = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
    r = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if (tp == 0 and fn == 0) else 0.0)
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    summary[name] = {
        "Precision": p,
        "Recall": r,
        "F1": f1,
        "FP": fp,
        "Missed": fn,
        "TP": tp,
        "TN": tn
    }
    print(f"{name:<26}: P={p*100:5.1f}%, R={r*100:5.1f}%, F1={f1*100:5.1f}%, FP={fp}, Missed={fn}")

print("\nFINAL SUMMARY:")
import json
print(json.dumps(summary, indent=2))
