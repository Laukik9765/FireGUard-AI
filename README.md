# FireGuard AI — README

> **FireGuard AI** is a vision-language-based fire situational-awareness system that uses NVIDIA LocateAnything for visual localization of fire, smoke, and living beings, combined with temporal analysis and a dedicated hazard assessment layer.

> ⚠️ **Research Prototype** — This is an academic/research prototype. It does not guarantee firefighter safety, determine whether a building is safe to enter, or confirm that a person is trapped. Smoke density values are internal engineering indicators only — not official fire-safety measurements. Absence of detection does not imply absence of danger.

---

## What it does

FireGuard AI continuously analyzes a live camera feed (or uploaded footage) and produces clear, actionable situational information:

```
FIRE        DETECTED
SMOKE       HIGH
DENSITY     78/100
COVERAGE    42%
PEOPLE      4
LIVESTOCK   3
HAZARD      HIGH
TREND       INCREASING
```

---

## Hardware Requirements

| Component | Minimum | Target |
|---|---|---|
| GPU | NVIDIA RTX 3050 (6 GB VRAM) | Any CUDA-capable GPU with ≥4 GB VRAM |
| RAM | 16 GB | 16+ GB |
| Python | 3.10+ | 3.11 |
| CUDA | 12.1 | 12.1+ |

> The full unquantized LocateAnything-3B model requires more than 6 GB VRAM.
> FireGuard uses **4-bit quantization** by default. Fall back to Q5 or Q4 if VRAM is insufficient.

---

## Setup

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd fireguard-ai
```

### 2. Create a Python environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install PyTorch (CUDA)

```bash
# For CUDA 12.1 (RTX 3050):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install FireGuard dependencies

```bash
pip install -r requirements.txt
```

### 5. Download LocateAnything-3B weights

> ⚠️ Verify the exact HuggingFace model ID before downloading.
> The current placeholder in `models/locate_anything/locate_anything_detector.py` is:
> `nvidia/locate-anything-3b` — confirm this on NVIDIA NGC / HuggingFace.

The model will download automatically on first use via HuggingFace Hub, or you can pre-download:

```bash
huggingface-cli download nvidia/locate-anything-3b
```

---

## Running

### Phase 1 — Baseline inference test (START HERE)

**Always run this first to verify the model loads on your RTX 3050.**

```bash
python tests/test_inference_baseline.py
```

This will:
- Confirm CUDA is available
- Load LocateAnything with 4-bit quantization
- Run all 4 queries on a blank frame
- Report latency and VRAM usage

### Running the dashboard (Mock mode — no GPU required)

```bash
python app.py --mock
```

Open: http://127.0.0.1:5000

### Running the dashboard (Real model — requires GPU + weights)

```bash
python app.py
```

### CLI options

```
python app.py --help

  --mock      Use mock detector (no GPU needed — for UI dev/testing)
  --host      Host to bind (default: 127.0.0.1)
  --port      Port to listen on (default: 5000)
  --debug     Enable Flask debug mode
```

---

## Running Tests

Unit tests do **not** require a GPU:

```bash
pytest tests/ -v --ignore=tests/test_inference_baseline.py
```

---

## Project Structure

```
FireGuard-AI/
│
├── app.py                          # Main entry point (Flask-SocketIO)
├── requirements.txt
├── README.md
│
├── models/
│   ├── locate_anything/
│   │   ├── locate_anything_detector.py   # LocateAnything-3B wrapper
│   │   └── mock_detector.py              # Mock detector for testing
│   └── fireguard/                        # Fine-tuned models (future)
│
├── core/
│   ├── detector.py           # Abstract Detector interface
│   ├── video_processor.py    # Live webcam pipeline
│   ├── image_processor.py    # Image/video upload analysis
│   ├── tracker.py            # Lightweight object tracker
│   ├── smoke_analyzer.py     # Smoke density / coverage / trend
│   ├── hazard_engine.py      # Temporal hazard aggregation
│   ├── event_manager.py      # Event lifecycle management
│   └── alert_manager.py      # State-transition-based alerts
│
├── dashboard/
│   ├── templates/index.html  # Control-room dashboard UI
│   └── static/
│       ├── css/style.css
│       └── js/dashboard.js
│
├── inputs/
│   ├── images/               # Uploaded images (never overwritten)
│   └── videos/               # Uploaded videos (never overwritten)
│
├── outputs/
│   ├── annotated/            # Annotated output images/videos
│   ├── reports/              # JSON analysis reports
│   └── events/               # Hazard event records
│
├── datasets/
│   ├── fire/
│   ├── smoke/
│   ├── people/
│   └── livestock/
│
└── tests/
    ├── test_smoke_analyzer.py
    ├── test_hazard_engine.py
    ├── test_mock_detector.py
    └── test_inference_baseline.py    # GPU required — run manually
```

---

## Architecture

FireGuard never imports LocateAnything directly in application code.
All model interactions go through the `Detector` interface:

```
FireGuard → Detector Interface → LocateAnythingDetector
                               → MockDetector (testing)
                               → FinetuedDetector (future)
```

This means the specialized FireGuard model (after fine-tuning) can replace the base model without touching the rest of the application.

---

## Development Roadmap

| Phase | Status | Focus |
|---|---|---|
| 1 | ⬜ | Baseline inference test on RTX 3050 |
| 2 | ⬜ | Fire/smoke/people/livestock prompt testing |
| 3 | ⬜ | Live webcam connection |
| 4 | ⬜ | Temporal processing (framing, tracking) |
| 5 | ⬜ | Smoke density analysis |
| 6 | ✅ | Dashboard (built) |
| 7 | ✅ | Image/video upload analysis |
| 8 | ⬜ | FireGuard dataset collection |
| 9 | ⬜ | Fine-tuning (cloud GPU) |
| 10 | ⬜ | Evaluation (base vs. fine-tuned) |
| 11 | ⬜ | Final demonstration |

---

## Known Limitations

- LocateAnything-3B is a **general-purpose** model — it is not specialized for fire/smoke detection out of the box. False positives on orange/yellow objects are likely before fine-tuning.
- Performance (FPS) has **not yet been benchmarked** on the RTX 3050. Never assume a specific FPS without measuring.
- Smoke density values are **internal engineering indicators** — not calibrated to any fire-safety standard.
- Fine-tuning on the RTX 3050 is not recommended — use a cloud GPU with more VRAM.
- The LocateAnything model ID (`nvidia/locate-anything-3b`) must be verified on HuggingFace/NGC before use.

---

## Safety Disclaimer

FireGuard AI must NOT be used to:
- Guarantee firefighter safety
- Determine whether a building is safe to enter
- Confirm that a person is trapped or unconscious
- Present smoke density as an official fire-safety measurement

This system provides **situational awareness and visual analysis** only.
It is an assistive technology research prototype, not a certified safety system.
