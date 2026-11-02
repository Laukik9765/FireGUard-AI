# FireGuard AI â€” Project Specification

> **Purpose of this document:** This is the complete build specification for the FireGuard AI project. It is intended to be handed to an autonomous coding agent (e.g. Antigravity) to scaffold, implement, and iterate on the full system. All requirements, constraints, and design rules below are mandatory unless explicitly marked optional.

---

## 1. Project Overview

**FireGuard AI** is a real-time, computer-vision-based fire safety and situational-awareness system. It helps firefighters understand what is happening inside or around a fire-affected area **before they enter**.

The goal is **not simply to detect fire**. The goal is to continuously analyze a live camera feed (or uploaded footage) and produce clear, actionable situational information:

- Whether fire is present, and where
- Whether smoke is present
- Smoke density / severity
- Percentage of the visible scene affected by smoke
- Whether smoke is increasing or decreasing over time
- Number of visible people, and their locations
- Number of visible livestock/animals, and their locations
- Overall hazard status

The system must also support **uploading existing images and video** for post-incident analysis, testing, demos, and historical CCTV review.

The project is built around **NVIDIA's LocateAnything-3B**, replacing a previous YOLO-based implementation.

### Core Principle

> "Don't just tell firefighters that there is a fire. Give them a visual understanding of what they may encounter before they enter."

Example of the kind of output the system should produce (not a generic alert):

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

## 2. Core Technology: NVIDIA LocateAnything-3B

LocateAnything-3B is the primary visual localization/grounding model.

**Workflow:**
```
Image + Natural Language Query â†’ Visual Localization â†’ Bounding Boxes / Points
```

Example queries:
- `"Locate visible fire"`
- `"Locate visible smoke"`
- `"Locate all people"`
- `"Locate all livestock"`

**Architecture:** Moon-ViT vision encoder + Qwen2.5 language decoder, connected via an MLP projector.

**Key feature â€” Parallel Box Decoding (PBD):** treats a bounding box as an atomic unit and predicts the full coordinate set together, rather than generating coordinate tokens sequentially.

**Capabilities:** general object detection, dense object detection, referring-expression grounding, point localization, text localization/OCR, GUI grounding, layout grounding.

**Inference modes (per NVIDIA docs):**
| Mode | Behavior |
|---|---|
| Fast Mode | Parallel decoding, optimized for throughput |
| Slow Mode | More stable, sequential decoding |
| Hybrid Mode | Fast by default, falls back to slow decoding when ambiguity/irregularity is detected |

### Important Limitation

LocateAnything is a **general-purpose** vision-language localization model â€” it is **not** a specialized fire-safety model out of the box. Do not assume it will reliably detect fire/smoke without domain adaptation.

**Development strategy:**
```
LocateAnything Base Model
   â†’ Baseline Testing
   â†’ FireGuard Dataset
   â†’ Domain-Specific Fine-Tuning
   â†’ FireGuard Specialized Model
```

Must be evaluated specifically for: fire detection, smoke detection, people detection, livestock detection, false positives, false negatives, dense smoke conditions, partial occlusion, low-light environments, indoor environments, outdoor environments.

---

## 3. Target Hardware

**Primary development machine:** NVIDIA RTX 3050 Laptop GPU â€” 6 GB VRAM

- The full unquantized LocateAnything-3B model should **not** be assumed to fit comfortably in 6 GB VRAM.
- Target a **quantized** model, initially **Q6_K**-style if practical.
- Fall back to **Q5_K** or **Q4_K** if VRAM/performance requires it.
- The laptop is for: local dev, testing, real-time inference, webcam testing, image/video analysis, dataset prep, evaluation.
- **Do not assume fine-tuning happens on the RTX 3050** â€” actual fine-tuning of the 3B model may require a larger cloud GPU.

---

## 4. Operating Modes

### A. Live Monitoring
Webcam (or CCTV-compatible stream; webcam for the current demo) is continuously monitored â€” no manual frame capture required.

```
Webcam â†’ Continuous Video Capture â†’ Frame Processing â†’ LocateAnything
       â†’ Fire / Smoke / People / Livestock â†’ Temporal Analysis
       â†’ FireGuard Hazard Engine â†’ Dashboard + Alerts
```

### B. Uploaded Footage Analysis

**Supported images:** JPG, JPEG, PNG, WebP
**Supported videos:** MP4, AVI, MOV, MKV

- User uploads image â†’ receives annotated analysis.
- User uploads video â†’ processed chronologically, producing a timeline, e.g.:

```
00:00  No hazard
00:05  Smoke detected
00:08  Smoke increasing
00:11  Fire detected
00:12  3 people detected
00:15  High hazard
```

Output should include: annotated video (where practical), detection timeline, fire events, smoke events, people counts, livestock counts, hazard changes.

**Original uploaded footage must never be overwritten.**

---

## 5. Real-Time Processing Strategy

Webcam may run at 15â€“30+ FPS, but full LocateAnything inference should **not** run on every frame (inefficient on 6 GB VRAM).

Pipeline components:
- Frame sampling
- AI inference on selected frames only
- Object persistence/tracking between inference frames
- Temporal analysis
- Efficient rendering

```
Camera: 30 FPS
Frame 1 â”€â”
Frame 2  â”‚
Frame 3  â”œâ”€â”€ Tracking / lightweight processing
Frame 4  â”‚
Frame 5 â”€â”˜   â†’ LocateAnything â†’ Updated detections
```

- Inference frequency must be **benchmarked on actual hardware**, never hard-coded from assumptions.
- Priority: **fast enough response while maintaining useful detection quality.**
- UI must never freeze during inference â€” use async/background processing.

---

## 6. Fire Detection

Prompted via natural language, e.g. `"Locate visible fire"` / `"Locate visible flames"`. Output as bounding boxes.

### Dataset Requirements

**Positive examples:** building fires, indoor fires, outdoor fires, small flames, large flames, vehicle fires, electrical/fire-like scenes, flames partially obscured by smoke.

**Negative examples (critical â€” prevents false positives):** orange objects, yellow objects, red objects, orange lights, lamps, reflections, sunlight, screens, clothing, decorative lights.

> âš ï¸ The previous FireGuard implementation suffered from false positives on orange/yellow objects being mistaken for fire. This must be specifically addressed in the new system.

---

## 7. Smoke Detection & Analysis

Smoke is queried via `"Locate visible smoke"`. The model returns a region â€” raw model output must **not** be treated directly as a "danger percentage." Build a dedicated **Smoke Analyzer** module.

### Smoke Density

Displayed as (example):
```
SMOKE       HIGH
Density:    78/100
Coverage:   42%
Trend:      Increasing
```

**Inputs the Smoke Analyzer should consider:**
- Smoke region size
- Coverage of the visible scene
- Visual opacity/intensity
- Relative density
- Persistence over multiple frames
- Expansion/change over time

```
Smoke Localization â†’ Smoke Region â†’ (Coverage + Visual Density + Persistence + Change Over Time)
   â†’ Smoke Density Index (0â€“100) â†’ Smoke Hazard Level
```

**Density levels (initial, to be calibrated later with FireGuard-specific footage):**
| Range | Level |
|---|---|
| 0â€“20 | None / Very Low |
| 21â€“40 | Low |
| 41â€“60 | Moderate |
| 61â€“80 | High |
| 81â€“100 | Critical |

> This numerical scale is an internal engineering indicator only â€” it must never be presented as an official fire-safety standard.

### Color Indicator

| Color | Level |
|---|---|
| Green | None / Very Low |
| Yellow | Low |
| Orange | Moderate |
| Red | High / Critical |

Always pair color with a text label (e.g. `â— HIGH â€” Density: 78/100`). Never rely on color alone.

### Smoke Coverage vs. Density

- **Density** â€” how visually dense does the smoke appear?
- **Coverage** â€” how much of the visible scene is affected (e.g. `Smoke Coverage: 42%`)?

Both are distinct and both must be shown.

### Smoke Trend

Track density over time across frames, e.g.:
```
Density: 20 â†’ 32 â†’ 48 â†’ 67 â†’ 78
Trend: â†‘ Increasing
```
States: **Increasing**, **Stable**, **Decreasing**.

---

## 8. People Detection

Query: `"Locate all people"`. Display count prominently (e.g. `PEOPLE: 4`), with bounding boxes on the live feed. Count updates as people enter/leave frame.

> Do not claim a detected person is definitely trapped, unconscious, or alive unless there is sufficient evidence. This is a visual detection/localization system, not a medical or rescue-status determination tool.

---

## 9. Livestock / Animal Detection

Detect living animals: cow, buffalo, goat, sheep, horse, dog, and other relevant animals. Display a simple count (e.g. `LIVESTOCK: 3`). Provides additional situational context for firefighters.

---

## 10. Temporal Hazard Engine

Do not trigger an emergency state from a single questionable frame â€” require temporal consistency.

**Example state progression:**
```
NORMAL â†’ SMOKE SUSPECTED â†’ SMOKE CONFIRMED â†’ FIRE SUSPECTED â†’ FIRE CONFIRMED â†’ EMERGENCY
```

Persistence thresholds (how many consecutive/recent frames are required to escalate) must be **configurable** and evaluated experimentally, not hard-coded from assumption.

### Overall Hazard Status

Levels: `NORMAL`, `LOW`, `MODERATE`, `HIGH`, `CRITICAL`

Considers: fire detection, smoke density, smoke coverage, smoke trend, detection persistence, number of people, number of livestock, other validated indicators.

> The hazard status is a **situational-awareness indicator only** â€” never a guarantee that an area is safe or unsafe to enter.

---

## 11. Alert System

On a confirmed fire/hazard event, notify the user clearly, e.g.:
```
ðŸš¨ FIRE DETECTED
Smoke: HIGH | Density: 78/100 | People: 4 | Livestock: 3 | Trend: Increasing
```

- Alerts must be clear and **non-spammy** â€” event-based, not per-frame.

```
Fire event begins â†’ Alert â†’ Event remains active (no repeat spam)
   â†’ Event resolved â†’ Ready for next event
```

---

## 12. UI / Dashboard Design Requirements

**This is a critical, non-negotiable section. The UI must look like real, professional monitoring software â€” not an "AI demo."**

### Explicitly Avoid (AI-slop indicators)
- Excessive gradients
- Glowing / neon effects
- Excessive or gratuitous animations
- Dozens of small decorative cards
- Large decorative illustrations or hero graphics
- Fake/marketing-style AI terminology ("Powered by Advanced Neural Intelligenceâ„¢", etc.)
- Unnecessary charts that don't inform a decision
- Excessive confidence-percentage displays
- Technical model internals shown on the main operational screen (model name, quantization, inference mode, etc. belong in Settings/diagnostics, not the main dashboard)

### Design Priorities (in order)
1. Readability
2. Speed of comprehension
3. Professional appearance
4. Low visual clutter
5. Clear information hierarchy
6. Emergency usability

### Style Direction
- Minimal, professional, control-room / monitoring-software aesthetic
- High contrast, easy to scan at a glance under stress
- Dark/light neutral theme as appropriate for a control-room context
- Consistent typography and spacing â€” no decorative flourishes
- Should look like it could realistically run in a real monitoring/dispatch environment

### Main Dashboard Must Answer, Within Seconds
- Is there fire?
- Is there smoke?
- How severe is the smoke?
- How much of the scene is affected?
- How many people are visible?
- Are animals/livestock visible?
- Is the situation getting worse?

### Recommended Layout

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ FIREGUARD AI                              â— SYSTEM ONLINEâ”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                            â”‚
â”‚                    LIVE CAMERA FEED                       â”‚
â”‚         Fire / Smoke / Person bounding boxes               â”‚
â”‚                                                            â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ FIRE           â”‚ SMOKE           â”‚ PEOPLE                 â”‚
â”‚ DETECTED       â”‚ HIGH            â”‚        4                â”‚
â”‚                â”‚ Density 78/100  â”‚                         â”‚
â”‚                â”‚ Coverage 42%    â”‚                         â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ HAZARD STATUS                                  HIGH        â”‚
â”‚ Smoke trend: â†‘ Increasing                                   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### Camera View Rules
The live feed is the primary visual element. Overlay only: fire bounding boxes, smoke region outline, person bounding boxes, livestock bounding boxes, and minimal labels. Do not obscure the video with excessive UI chrome.

### Navigation (keep minimal)

```
FireGuard AI
â”œâ”€â”€ Live Monitoring      â€” primary emergency screen
â”œâ”€â”€ Analyze Footage      â€” upload image/video
â”œâ”€â”€ Events                â€” history of previous detections/timelines
â””â”€â”€ Settings              â€” camera, processing, alerts, app config
```

---

## 13. Software Architecture

```
FireGuard-AI/
â”‚
â”œâ”€â”€ app.py
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ README.md
â”‚
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ locate_anything/
â”‚   â””â”€â”€ fireguard/
â”‚
â”œâ”€â”€ core/
â”‚   â”œâ”€â”€ detector.py           # Detector interface / abstraction
â”‚   â”œâ”€â”€ video_processor.py
â”‚   â”œâ”€â”€ image_processor.py
â”‚   â”œâ”€â”€ tracker.py
â”‚   â”œâ”€â”€ smoke_analyzer.py
â”‚   â”œâ”€â”€ hazard_engine.py
â”‚   â”œâ”€â”€ event_manager.py
â”‚   â””â”€â”€ alert_manager.py
â”‚
â”œâ”€â”€ inputs/
â”‚   â”œâ”€â”€ images/
â”‚   â””â”€â”€ videos/
â”‚
â”œâ”€â”€ outputs/
â”‚   â”œâ”€â”€ annotated/
â”‚   â”œâ”€â”€ reports/
â”‚   â””â”€â”€ events/
â”‚
â”œâ”€â”€ dashboard/
â”‚
â”œâ”€â”€ datasets/
â”‚   â”œâ”€â”€ fire/
â”‚   â”œâ”€â”€ smoke/
â”‚   â”œâ”€â”€ people/
â”‚   â””â”€â”€ livestock/
â”‚
â””â”€â”€ tests/
```

### Detector Abstraction

Keep LocateAnything behind a `Detector` interface so the rest of the application never depends directly on its implementation details:

```
FireGuard â†’ Detector Interface â†’ LocateAnything
```

This means a specialized/fine-tuned model can later replace or supplement LocateAnything for a specific task (fire, smoke, people, livestock) without rewriting the application.

---

## 14. Performance Requirements

- Never block the UI during inference.
- Use background/asynchronous processing where appropriate.
- Avoid running heavy inference unnecessarily.
- Process selected frames, not every camera frame.
- Use lightweight tracking between expensive model calls.
- Keep the model resident in memory â€” avoid repeated reloads.
- Use GPU acceleration; monitor VRAM usage.
- Benchmark actual FPS/latency on the RTX 3050 â€” **never claim a specific FPS without measuring it.**
- Priority: **low response latency** while maintaining acceptable detection quality.

---

## 15. Dataset Strategy

Each training example:
```
Image + Natural Language Query + Bounding Box / Spatial Annotation
```

Examples:
```
fire_001.jpg        "Locate visible fire"      â†’ bounding box
smoke_001.jpg       "Locate visible smoke"     â†’ bounding box
scene_001.jpg       "Locate all people"        â†’ multiple bounding boxes
farm_fire_001.jpg   "Locate livestock"         â†’ multiple bounding boxes
```

### Negative Dataset (essential for reducing false positives)

**Fire negatives:** orange objects, yellow objects, red objects, lamps, sunlight, reflections, decorative lights, screens, clothing.

**Smoke negatives:** fog, clouds, steam, dust, shadows, atmospheric haze.

---

## 16. Fine-Tuning Strategy

Do **not** fine-tune immediately â€” establish a baseline first.

```
LocateAnything base model â†’ FireGuard test dataset â†’ Measure performance
```

Then fine-tune:
```
LocateAnything-3B â†’ FireGuard-specific dataset â†’ Domain-specific fine-tuning â†’ FireGuard model
```

Compare base vs. fine-tuned model on:
- Precision, Recall, F1
- Detection accuracy
- Localization quality
- False-positive rate
- False-negative rate
- Inference latency
- FPS

---

## 17. Project Positioning & Safety Disclaimers

**Describe the project as:**
> A vision-language-based fire situational-awareness system that uses NVIDIA LocateAnything for visual localization of fire, smoke, and living beings, combined with temporal analysis and a dedicated hazard assessment layer.

**Do not describe it merely as** "Fire Detection System" â€” it is broader than detection.

### This is an academic/research prototype. It must NOT claim to:
- Guarantee firefighter safety
- Guarantee that a person is trapped
- Determine that a building is safe/unsafe to enter
- Present smoke density measurements as official fire-safety measurements
- Have perfect detection
- Imply that absence of detection means absence of danger

The dashboard provides **situational awareness and visual analysis** â€” it should be presented throughout as an assistive technology prototype, not a certified safety system.

---

## 18. Development Roadmap

| Phase | Focus |
|---|---|
| 1 | Model Setup â€” install LocateAnything, configure quantized model + CUDA, verify RTX 3050 inference, simple image tests |
| 2 | Basic Detection â€” test fire/smoke/people/livestock prompts |
| 3 | Webcam â€” connect webcam, continuous capture, real-time inference, basic bounding boxes |
| 4 | Temporal Processing â€” frame sampling, tracking, persistence, event detection, hazard states |
| 5 | Smoke Analysis â€” localization region, coverage, density index, trend, hazard level |
| 6 | Dashboard â€” build the clean, professional live dashboard |
| 7 | Image/Video Upload â€” image analysis, video analysis, timeline, annotated output |
| 8 | Dataset â€” collect FireGuard-specific images/videos |
| 9 | Fine-Tuning â€” fine-tune LocateAnything on FireGuard data (cloud GPU) |
| 10 | Evaluation â€” compare base vs. fine-tuned model |
| 11 | Final Demonstration â€” full pipeline: live webcam, fire, smoke, density, trend, people, livestock, hazard, alerts, uploaded video analysis |

**Development should proceed incrementally, starting with a working LocateAnything inference test on the RTX 3050 before building the full pipeline.**

---

## 19. Final Product Vision

The finished system should feel like a **real monitoring product**, not a collection of AI demos.

**Live flow:**
```
Open FireGuard â†’ Connect Camera â†’ Live Monitoring
   â†’ System continuously watches
   â†’ Fire / smoke / people / livestock detected
   â†’ Information updated in real time
   â†’ Hazard status changes when necessary
   â†’ User receives a clear alert
```

**Uploaded footage flow:**
```
Upload Video â†’ Analyze â†’ Detection Timeline â†’ Smoke Progression
   â†’ Fire Events â†’ People/Livestock Count â†’ Final Incident Summary
```

The system should always prioritize **clarity over complexity**. The UI must be minimal, professional, responsive, high-contrast, easy to scan, consistent, free of unnecessary AI visual effects, and suitable for emergency/control-room use.

---

## 20. Build Instructions Summary (for the implementing agent)

1. Do **not** build a generic "AI dashboard." Build professional monitoring software.
2. No excessive gradients, glowing effects, animations, or decorative AI visuals.
3. Keep the UI clean, professional, and software-like at all times.
4. Prioritize responsiveness â€” never block the UI during inference.
5. Keep LocateAnything fully isolated behind a `Detector` interface.
6. Do not assume the base LocateAnything model is already optimized for fire/smoke.
7. Build the system so it can be fine-tuned later without architectural rework.
8. Support both live webcam monitoring and uploaded image/video analysis.
9. Never overwrite original uploaded media.
10. Implement smoke density as a dedicated FireGuard analysis feature â€” not a raw LocateAnything output.
11. Keep the live dashboard focused on: fire, smoke, people, hazard status.
12. Avoid unnecessary technical/model information on the main screen (move to Settings/diagnostics).
13. Build modularly so components can be tested independently.
14. Never hard-code performance claims (e.g. FPS) without benchmarking on the RTX 3050.
15. Never present experimental smoke-density values as official safety standards.
16. Ensure the application is usable as a research prototype on an RTX 3050 (6 GB VRAM).
17. Optimize for fast response and stable continuous monitoring.
18. Keep code readable and maintainable.
19. Document all setup steps, model requirements, hardware requirements, and known limitations in the README.
20. Start with a working LocateAnything inference test on the RTX 3050 before implementing the full pipeline.

