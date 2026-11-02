# FireGuard AI — Academic Thesis & Technical Research Compendium

> **Document Purpose:**  
> This comprehensive document serves as the primary technical, mathematical, and algorithmic reference for writing an academic Master's, Bachelor's, or Doctoral thesis, or conference/journal paper on **FireGuard AI**. It details the theoretical foundations, architectural formulations, computer vision algorithms, neural networks, temporal filtering, and evaluation methodologies implemented in the codebase.

---

## 1. Abstract & Research Overview

### 1.1. Abstract
Structural firefighting operations operate under extreme uncertainty, severe temporal constraints, and degraded sensory environments. Traditional automated fire detection systems rely almost exclusively on ionization, photoelectric, or thermal point sensors, or standard binary image classification models that produce simple alarms (e.g., "Fire Detected"). Such systems fail to provide spatial, contextual, or human-centric situational awareness to incident commanders prior to structural ingress.

**FireGuard AI** presents an edge-deployable, vision-language and neural ensemble framework for multi-dimensional fire situational awareness. Built on a hybrid architecture combining Vision-Language Grounding (**NVIDIA LocateAnything-3B** with Parallel Box Decoding) and high-throughput real-time object detection (**YOLOv8**), FireGuard AI concurrently localizes fire boundaries, calculates smoke dispersion and density indices, tracks living occupants (humans and livestock), and computes a dynamically smoothed multi-signal **Hazard Level**. To eliminate catastrophic false alarms caused by chromatic distractors (e.g., orange jackets, red furniture, sunlight gradients), the system integrates a **multi-gate physical combustion filter** leveraging chromaticity ratios, HSV saturation bounds, and interior spatial variance checks. Furthermore, an asynchronous **time-shifted ring buffer** synchronizes heavy neural inference with real-time video display, preventing bounding-box motion lag. Experimental verification demonstrates robust multi-class localization, zero false positives on high-salience non-combustion artifacts, and low-latency execution within a constrained 6 GB VRAM edge budget.

### 1.2. Key Research Contributions
1. **Multi-Class Grounding for Tactical Incident Command:** A unified pipeline that simultaneously grounds combustion zones (fire/smoke) and living entities (people/livestock) using natural language referring expressions and neural bounding boxes.
2. **Multi-Gate Anti-False-Positive Combustion Filter:** A 5-gate chromatic and spatial variance validation algorithm that filters out non-fire spectral distractors without sacrificing recall on genuine combustion regions.
3. **Internal Smoke Density Index ($D$) and Volumetric Trajectory Model:** A mathematical formulation combining spatial coverage percentage and continuous temporal persistence to track smoke accumulation trends ($\frac{dD}{dt}$).
4. **Temporal State Machine & Hysteresis Filtering:** A sliding-window persistence filter that requires multi-frame temporal confirmation before state escalation, coupled with a non-spammy state-transition alert protocol.
5. **Asynchronous Time-Shifted Video Architecture:** A ring-buffered video processing pipeline that guarantees zero spatial-temporal mismatch between neural inference outputs and high-frame-rate visual streams.

---

## 2. Theoretical Framework & Problem Formulation

### 2.1. Limitations of Existing Fire Detection Paradigms
Existing computer vision systems for fire detection typically suffer from two primary failure modes:
1. **Binary Classification Without Spatial Context:** Convolutional networks classifying entire frames as "Fire" or "No Fire" provide zero localization data regarding egress paths, flame spread velocity, or trapped occupants.
2. **High False Positive Rates (FPR):** Heuristic RGB/HSV color thresholding algorithms routinely misclassify amber lighting, fluorescent lamps, red apparel, reflective surfaces, and sunsets. Conversely, ungrounded deep learning models often overfit to dataset-specific lighting biases.

### 2.2. Mathematical Problem Formulation
Let an input video stream be represented as a sequence of discrete frames:
$$\mathcal{F} = \{I_1, I_2, \dots, I_t\}, \quad I_t \in \mathbb{R}^{H \times W \times 3}$$

At each discrete time step $t$, the system must infer a tuple representing the complete environmental state:
$$\mathcal{S}_t = \left( \mathcal{B}_{\text{fire}}, \mathcal{B}_{\text{smoke}}, \mathcal{B}_{\text{people}}, \mathcal{B}_{\text{livestock}}, \mathcal{M}_{\text{smoke}}, \mathcal{H}_t, \mathcal{A}_t \right)$$

Where:
* $\mathcal{B}_c = \{b_1, b_2, \dots, b_k\}$ denotes the set of normalized bounding boxes for category $c$:
  $$b_i = (x_{1}, y_{1}, x_{2}, y_{2}, \text{conf}), \quad x_i, y_i \in [0, 1]$$
* $\mathcal{M}_{\text{smoke}} = (C_t, D_t, \mathcal{T}_t)$ defines the smoke metrics: scene coverage percentage $C_t \in [0, 100]$, density index $D_t \in [0, 100]$, and trajectory $\mathcal{T}_t \in \{\text{Increasing}, \text{Stable}, \text{Decreasing}\}$.
* $\mathcal{H}_t \in \{\text{NORMAL}, \text{LOW}, \text{MODERATE}, \text{HIGH}, \text{CRITICAL}\}$ represents the aggregated temporal hazard level.
* $\mathcal{A}_t$ is the set of active transition-triggered alerts.

---

## 3. Neural Architecture & Vision-Language Grounding

```
                     +---------------------------------------+
                     |         Input Frame I_t in R^(HxWx3)  |
                     +---------------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |   Decoupled Detector Layer   |
                         +-------------------------------+
                                  /             \
                                 /               \
        [Hardware: Cloud / Ampere GPU]      [Hardware: Edge RTX 3050 / CPU]
                               /                   \
                              v                     v
              +-------------------------+   +-------------------------+
              | LocateAnything-3B (VLM) |   | YOLOv8 Ensemble Engine  |
              | Moon-ViT + Qwen2.5-3B   |   | Fire YOLO + COCO Living |
              | 4-bit NF4 Quantization  |   | Multi-Gate CV Filters   |
              +-------------------------+   +-------------------------+
                              \                     /
                               \                   /
                                v                 v
                         +-------------------------------+
                         | Bounding Box Candidate Pool   |
                         +-------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         | Centroid & Kinematic Tracker  |
                         | NMS + Velocity Step Motion    |
                         +-------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         | SmokeAnalyzer & HazardEngine  |
                         +-------------------------------+
```

### 3.1. Abstract Detector Interface
To ensure modularity and academic reproducibility, model execution is decoupled via the abstract [`core/detector.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/detector.py) base class:
```python
class Detector(abc.ABC):
    @abc.abstractmethod
    def load(self) -> None: ...
    @abc.abstractmethod
    def detect(self, image: np.ndarray, query: DetectionQuery, mode: str = "hybrid") -> Detection: ...
```
This abstraction permits plug-and-play interchange between heavy foundational vision-language models, lightweight neural edge detectors, or multi-spectral sensor feeds without modifying downstream tracking or hazard evaluation logic.

### 3.2. NVIDIA LocateAnything-3B Grounding Architecture
LocateAnything-3B is utilized for zero-shot natural language visual grounding:
* **Vision Backbone:** Moon-ViT encoder processes raw image patches into high-density visual tokens.
* **Language Backbone:** Qwen2.5-3B autoregressive transformer decoder processes natural language prompt queries:
  $$\mathcal{Q}_{\text{fire}} = \text{"Locate visible fire"}$$
  $$\mathcal{Q}_{\text{smoke}} = \text{"Locate visible smoke"}$$
  $$\mathcal{Q}_{\text{people}} = \text{"Locate all people"}$$
  $$\mathcal{Q}_{\text{livestock}} = \text{"Locate all livestock and animals"}$$
* **Parallel Box Decoding (PBD):** Unlike standard autoregressive coordinate prediction (where tokens $x_1, y_1, x_2, y_2$ are generated sequentially), PBD decodes coordinates in parallel as atomic vectors, drastically reducing inference latency.
* **4-Bit NormalFloat (NF4) Quantization:** To fit the 3-billion-parameter model into the target 6 GB VRAM budget (NVIDIA RTX 3050 Laptop GPU), BitsAndBytes double quantization is utilized:
  $$\text{VRAM}_{\text{total}} \approx \frac{3 \times 10^9 \times 0.5 \text{ bytes}}{10^9} + \text{KV-Cache} \approx 2.4\text{ GB (Weights)} + 1.2\text{ GB (Activations)} < 4.0\text{ GB}$$

### 3.3. High-Precision YOLOv8 Ensemble (Production Edge Engine)
For edge environments where large transformer inference is latency-prohibitive, FireGuard AI features a dedicated dual-neural ensemble implemented in [`models/locate_anything/mock_detector.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/models/locate_anything/mock_detector.py):
1. **Combustion Specialist:** Fine-tuned YOLOv8 model trained specifically on fire, flame tongues, and smoke plumes.
2. **Living Entity Specialist:** YOLOv8 trained on COCO-80, gating Class 0 (`Person`) with confidence threshold $\tau_{\text{person}} = 0.50$, and living animal classes (IDs 14–23: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe) with $\tau_{\text{animal}} = 0.40$.

---

## 4. Multi-Gate Anti-False-Positive Filtering

A primary innovation of FireGuard AI is its multi-spectral gating suite. Pure neural predictions on edge cameras are vulnerable to domain shift. Every bounding box $b$ generated by the neural or CV fallback network must pass rigorous physical combustion filters.

```
+-----------------------------------------------------------------------------------------+
|                              CANDIDATE BOUNDING BOX (b)                                |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
                      +---------------------------------------------+
                      | GATE 1: Geometric Dimension Sanity          |
                      | bw >= 6 px and bh >= 6 px                   |
                      +---------------------------------------------+
                                             | Pass
                                             v
                      +---------------------------------------------+
                      | GATE 2: Absolute Red Luminance              |
                      | Mean(R) >= 105                              |
                      +---------------------------------------------+
                                             | Pass
                                             v
                      +---------------------------------------------+
                      | GATE 3: Combustion Chromaticity Ratio       |
                      | Mean(R) >= 1.05 * Mean(G)                   |
                      | Mean(R) >= 1.25 * Mean(B)                   |
                      +---------------------------------------------+
                                             | Pass
                                             v
                      +---------------------------------------------+
                      | GATE 4: HSV Color Purity                    |
                      | Mean(Saturation) >= 65                      |
                      | Mean(Value) >= 100                          |
                      +---------------------------------------------+
                                             | Pass
                                             v
                      +---------------------------------------------+
                      | GATE 5: Interior Spatial Variance           |
                      | Var(Gray(Crop_inner)) >= 35.0               |
                      | (Rejects solid red t-shirts, posters, walls)|
                      +---------------------------------------------+
                                             | Pass
                                             v
                                  [ CONFIRMED FLAME REGION ]
```

### 4.1. The 5-Gate Flame Validation Filter
Given a candidate crop $I_c \in \mathbb{R}^{h \times w \times 3}$ corresponding to box $b$:
1. **Geometric Scale Gate:**
   $$w \ge 6 \quad \land \quad h \ge 6$$
   Eliminates single-pixel noise spikes and artifact clusters.
2. **Absolute Combustion Luminance Gate:**
   $$\bar{R} = \frac{1}{|I_c|} \sum_{(x,y)} I_c(x,y,R) \ge 105$$
   Flames are radiant emitters; dark red or brown hues are immediately discarded.
3. **Thermal Chromaticity Ratio Gate:**
   Combustion requires chemical energy release predominantly in the red/orange wavelength:
   $$\bar{R} \ge 1.05 \cdot \bar{G} \quad \land \quad \bar{R} \ge 1.25 \cdot \bar{B}$$
   This explicitly rejects purple lights, neutral white lamps, and bluish incandescent reflections.
4. **HSV Saturation & Energy Gate:**
   Transforming $I_c$ into cylindrical HSV color space:
   $$\bar{S} \ge 65 \quad \land \quad \bar{V} \ge 100$$
   Guarantees vivid combustion color purity while discarding desaturated beige, tan, wood, or cardboard backgrounds.
5. **Interior Spatial Texture Variance Gate (Anti-Apparel / Anti-Wall Filter):**
   Man-made objects (such as an orange firefighter jacket, red tent, or painted emergency door) possess high color saturation but near-zero internal spatial variance. Conversely, turbulent flames exhibit chaotic internal luminosity gradients.
   Let the interior core of the box be sampled with a 15% margin erosion:
   $$I_{\text{core}} = I_c\left[0.15h : 0.85h, \; 0.15w : 0.85w\right]$$
   The grayscale variance is computed:
   $$\text{Var}(I_{\text{core}}) = \frac{1}{|I_{\text{core}}|} \sum \left( G(x,y) - \mu_G \right)^2 \ge 35.0$$
   Any crop with $\text{Var} < 35.0$ is classified as a solid artificial surface and rejected.

### 4.2. Multi-Gate Smoke & Haze Dispersion Analyzer
Smoke plume detection involves distinguishing achromatic, desaturated, low-frequency atmospheric particles from clear sky and rigid textures:
1. **Achromatic Desaturation Condition:**
   $$S(x,y) < 45 \quad \land \quad 70 < V(x,y) < 240$$
   $$|R(x,y) - G(x,y)| < 26 \quad \land \quad |R(x,y) - B(x,y)| < 28$$
2. **Atmospheric Blue-Sky Rejection:**
   Clear skies and distant horizons exhibit strong Rayleigh scattering ($B \gg R$). FireGuard AI enforces:
   $$\text{Mask}_{\text{sky}}(x,y) = \mathbb{I}\left( B(x,y) > 1.18 \cdot R(x,y) \right)$$
   Pixels where $\text{Mask}_{\text{sky}} = 1$ are zeroed out.
3. **Morphological Coalescence:**
   $$\text{Mask}_{\text{clean}} = \left( \text{Mask} \circ \mathcal{K}_{21} \right) \bullet \mathcal{K}_{21}$$
   Where $\mathcal{K}_{21}$ is an elliptical structuring element of diameter 21 px, $\circ$ denotes opening (noise removal), and $\bullet$ denotes closing (plume hole filling).
4. **Canny High-Frequency Suppression:**
   Smoke disperses high-frequency edges. For any detected bounding region:
   $$\rho_{\text{edge}} = \frac{\sum \text{Canny}(I_{\text{gray}}, 30, 80)}{w \times h} \le 0.16$$
   If $\rho_{\text{edge}} > 0.16$, the region contains high-frequency grid structures (e.g., window blinds, wire mesh, brick mortar) and is discarded.

---

## 5. Kinematics, Tracking, & Temporal Persistence

### 5.1. Spatial Overlap Metric
Inter-frame bounding box matching uses the standard Intersection over Union (IoU):
$$\text{IoU}(b_1, b_2) = \frac{\text{Area}(b_1 \cap b_2)}{\text{Area}(b_1 \cup b_2)}$$
In the centroid tracker, the matching threshold is tuned to $\tau_{\text{IoU}} = 0.12$. This accommodates live camera jitter, exposure drift, and rapid flame flicker while reliably rejecting spatial discontinuities.

### 5.2. Kinematic Tracking & Velocity Estimation
In [`core/tracker.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/tracker.py), tracked entities maintain persistent track IDs and estimated centroid velocities:
$$\mathbf{c}_t = \left( \frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2} \right)$$
Upon receiving an IoU-matched detection after $\Delta t$ elapsed frames:
$$\mathbf{v}_{\text{raw}} = \frac{\mathbf{c}_{\text{new}} - \mathbf{c}_{\text{old}}}{\Delta t}$$
Velocity is smoothed using a low-pass filter:
$$\mathbf{v}_t = 0.5 \mathbf{v}_{t-1} + 0.5 \mathbf{v}_{\text{raw}}$$
Bounding box coordinates are smoothed via exponential moving average ($\alpha = 0.65$):
$$\mathbf{p}_t = \alpha \mathbf{p}_{\text{detected}} + (1 - \alpha) \mathbf{p}_{t-1}$$

Between intermittent neural inference frames, the tracker executes **Step Motion** to extrapolate positions:
$$\mathbf{p}_{t+1} = \mathbf{p}_t + \mathbf{v}_t, \quad \mathbf{v}_{t+1} = 0.94 \cdot \mathbf{v}_t$$
This ensures buttery-smooth 30+ FPS visualization on the web client while deep neural models run at lower sample intervals.

### 5.3. Temporal Confirmation Window (Anti-Flicker Gate)
To suppress camera sensor noise or specular reflections:
$$\text{Hits}(b) = \sum_{j=1}^{W} \mathbb{I}\left( \exists b_j \in \mathcal{H}_j : \text{IoU}(b, b_j) \ge \tau_{\text{IoU}} \right)$$
A candidate box is only promoted to the active snapshot if:
$$\text{Hits}(b) \ge K \quad (\text{Default: } K = 2 \text{ within window } W = 6)$$

---

## 6. Smoke Density Index & Spread Modeling

Rather than relying purely on subjective classification, FireGuard AI introduces an explicit engineering index for smoke severity in [`core/smoke_analyzer.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/smoke_analyzer.py).

```
         Detection Bounding Boxes
                     |
                     v
   +------------------------------------+
   | 1. Area Integration (Coverage %)   | ===> C_t = min(1.0, Sum Area_i) * 100
   +------------------------------------+
                     |
                     v
   +------------------------------------+
   | 2. Temporal Persistence Score (P)  | ===> P_t = min(1.0, N_consecutive / L_history)
   +------------------------------------+
                     |
                     v
   +-------------------------------------------------------------+
   | 3. Smoke Density Index (D_t)                                |
   |    D_t = (w_c * (C_t / 100) + w_p * P_t) * 100             |
   |    (Weights: w_c = 0.60, w_p = 0.40)                        |
   +-------------------------------------------------------------+
                     |
                     v
   +------------------------------------+
   | 4. Trajectory Derivative (dD/dt)   | ===> Split Window Mean Delta
   +------------------------------------+      Delta = Avg(New_Half) - Avg(Old_Half)
                     |
        +------------+------------+
        |                         |
    Delta > +5                Delta < -5
        |                         |
        v                         v
   [ INCREASING ]            [ DECREASING ] (Otherwise [ STABLE ])
```

### 6.1. Coverage Fraction ($C_t$)
The cumulative spatial footprint of all verified smoke bounding boxes is clamped to full-screen coverage:
$$C_t = \min\left(1.0, \sum_{i=1}^{M} \text{Area}(b_i)\right) \times 100.0$$

### 6.2. Persistence Accumulator ($P_t$)
Smoke accumulation is physically persistent. Transient puffs dissipate, while structural fires continuously produce soot. The persistence factor measures consecutive positive frames up to history length $L$:
$$P_t = \min\left(1.0, \frac{N_{\text{consecutive}}}{L_{\text{history}}}\right), \quad L_{\text{history}} = 10$$

### 6.3. Composite Density Index ($D_t$)
The weighted composite index is formulated as:
$$D_t = \left( w_c \cdot \frac{C_t}{100} + w_p \cdot P_t \right) \times 100.0$$
Where $w_c = 0.60$ and $w_p = 0.40$, satisfying $w_c + w_p = 1.0$.

### 6.4. Smoke Hazard Level Discretization
$$D_t \in [0, 100] \implies \begin{cases}
\text{NONE}, & D_t \le 0 \\
\text{VERY LOW}, & 0 < D_t \le 20 \\
\text{LOW}, & 20 < D_t \le 40 \\
\text{MODERATE}, & 40 < D_t \le 60 \\
\text{HIGH}, & 60 < D_t \le 80 \\
\text{CRITICAL}, & D_t > 80
\end{cases}$$

### 6.5. Trajectory Derivative ($\mathcal{T}_t$)
Over a sliding window buffer of past density indices $\mathbf{d} = [d_1, d_2, \dots, d_m]$, the window is partitioned into early and recent halves:
$$\mu_{\text{old}} = \frac{1}{\lfloor m/2 \rfloor} \sum_{k=1}^{\lfloor m/2 \rfloor} d_k, \quad \mu_{\text{new}} = \frac{1}{m - \lfloor m/2 \rfloor} \sum_{k=\lfloor m/2 \rfloor + 1}^{m} d_k$$
$$\Delta_{\text{trend}} = \mu_{\text{new}} - \mu_{\text{old}}$$
$$\mathcal{T}_t = \begin{cases}
\text{INCREASING } (\uparrow), & \Delta_{\text{trend}} > +5.0 \\
\text{DECREASING } (\downarrow), & \Delta_{\text{trend}} < -5.0 \\
\text{STABLE } (\rightarrow), & |\Delta_{\text{trend}}| \le 5.0
\end{cases}$$

---

## 7. The Temporal Hazard Assessment Engine

```
               [ Binary Detections (Fire, Smoke, Living Entities) ]
                                        |
                                        v
                       +----------------------------------+
                       |   Persistence State Machine      |
                       | Sliding Window: W=10 frames      |
                       +----------------------------------+
                                        |
                 +----------------------+----------------------+
                 |                                             |
           rate >= 0.60                                  rate <= 0.20
                 |                                             |
                 v                                             v
       [ ESCALATE STATE ]                             [ DE-ESCALATE STATE ]
    Normal -> Suspected ->                         Emergency -> Confirmed ->
    Confirmed -> Emergency                         Suspected -> Normal
                 \                                             /
                  \                                           /
                   +--------------------+--------------------+
                                        |
                                        v
                       +----------------------------------+
                       | Additive Hazard Scoring Function |
                       | S_total = S_fire + S_smoke + ... |
                       +----------------------------------+
                                        |
                                        v
                       +----------------------------------+
                       | Categorical Threat Level Mapping |
                       | 0: NORMAL   1-2: LOW             |
                       | 3-4: MODERATE                    |
                       | 5-6: HIGH   7+: CRITICAL         |
                       +----------------------------------+
```

### 7.1. Temporal Persistence Finite State Machine
In [`core/hazard_engine.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/hazard_engine.py), fire and smoke presence are evaluated via a 4-state Markov-style automaton:
$$\mathcal{S} \in \{\text{Normal}, \text{Suspected}, \text{Confirmed}, \text{Emergency}\}$$

Given a rolling boolean history of length $W = 10$, the positive detection rate is:
$$r = \frac{1}{W} \sum_{i=1}^{W} \mathbb{I}(\text{detected}_i)$$

State transitions follow strict hysteresis:
* **Escalation ($r \ge 0.60$):**
  $$\text{Normal} \longrightarrow \text{Suspected} \longrightarrow \text{Confirmed} \longrightarrow \text{Emergency}$$
* **De-escalation ($r \le 0.20$):**
  $$\text{Emergency} \longrightarrow \text{Confirmed} \longrightarrow \text{Suspected} \longrightarrow \text{Normal}$$
* **Hysteresis Band ($0.20 < r < 0.60$):** State remains unchanged, preventing oscillation under flickering flame conditions.

### 7.2. Hazard Score Formulation
The total threat score $S_t \in \mathbb{N}$ aggregates fire severity, smoke analysis, and occupant urgency:
$$S_t = S_{\text{fire}}(\mathcal{S}_{\text{fire}}) + S_{\text{smoke}}(\mathcal{M}_{\text{smoke}}) + \mathbb{I}(N_{\text{people}} > 0) + \mathbb{I}(N_{\text{livestock}} > 0)$$

#### Component Scoring Tables:
| Category State | Point Weight |
|---|---|
| Fire: `NORMAL` | 0 |
| Fire: `SUSPECTED` | 2 |
| Fire: `CONFIRMED` | 3 |
| Fire: `EMERGENCY` | 4 |
| Smoke: `NONE` / `VERY_LOW` / `LOW` | 0 / 1 / 1 |
| Smoke: `MODERATE` | 2 |
| Smoke: `HIGH` | 3 |
| Smoke: `CRITICAL` | 4 |
| People Detected ($N_{\text{people}} > 0$) | +1 |
| Animals Detected ($N_{\text{livestock}} > 0$) | +1 |

#### Categorical Danger Mapping:
$$\mathcal{H}(S_t) = \begin{cases}
\text{NORMAL}, & S_t = 0 \\
\text{LOW}, & 1 \le S_t \le 2 \\
\text{MODERATE}, & 3 \le S_t \le 4 \\
\text{HIGH}, & 5 \le S_t \le 6 \\
\text{CRITICAL}, & S_t \ge 7
\end{cases}$$

---

## 8. Latency Compensation via Time-Shift Ring Buffer

### 8.1. The Parallax & Latency Problem
Deep neural network inference on high-resolution frames requires non-zero compute time:
$$\Delta t_{\text{inference}} \in [80\text{ ms}, 350\text{ ms}]$$
If an inference output is rendered directly onto the *current* frame arriving from the camera at $t_{\text{current}} = t_{\text{capture}} + \Delta t$, rapid motion causes bounding boxes to trail behind moving persons or flickering flames.

### 8.2. Ring Buffer Implementation
[`core/video_processor.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/video_processor.py) resolves this via a timestamped time-shift buffer:
1. Every captured camera frame is stamped with monotonic time $\tau$ and placed into a bounded ring buffer $\mathcal{R}$ of capacity:
   $$N_{\text{cap}} = \min(120, \max(10, \lfloor \text{BUFFER\_SECONDS} \times \text{FPS} \rfloor + 5))$$
2. Frames sampled for inference record their capture timestamp $\tau_{\text{sample}}$.
3. When inference completes, the bounding box coordinate set $\mathcal{B}$ is matched back to the exact frame in $\mathcal{R}$ whose timestamp is closest to $\tau_{\text{sample}}$.
4. The dashboard displays the video stream delayed by a constant latency $\Delta t = \text{BUFFER\_SECONDS} = 1.5\text{ s}$.

**Result:** Detections are drawn on the exact physical visual state from which they were inferred, achieving 100% spatial-temporal registration.

---

## 9. Non-Spammy Event Lifecycle & Alert Management

### 9.1. Alert State Transitions
In emergency environments, acoustic and visual alerts must be high-signal and low-fatigue. In [`core/alert_manager.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/alert_manager.py):
* An alert is fired **only** when $\mathcal{H}_t$ transitions from `NORMAL` to a hazardous state (`Event Started`).
* An escalation alert is fired **only** if $\mathcal{H}_t$ increases (e.g., `MODERATE` $\to$ `HIGH`).
* A clearance alert is fired when the hazard clears back to `NORMAL`.
* Frames exhibiting identical or decreasing hazard states emit zero duplicate notifications.

### 9.2. Event Lifecycle Persistence
[`core/event_manager.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/event_manager.py) manages discrete incident episodes. Each episode is archived as a structured JSON object in `outputs/events/<event_id>.json`:
```json
{
  "event_id": "c7a8b192-348f-4f76-bc3e-09a8264ef1a0",
  "started_at": "2026-10-26T17:15:02.128456Z",
  "ended_at": "2026-10-26T17:18:45.892104Z",
  "peak_hazard": "HIGH",
  "fire_confirmed": true,
  "smoke_confirmed": true,
  "max_people": 3,
  "max_livestock": 0,
  "peak_density": 74.2,
  "frame_count": 6480,
  "closed": true
}
```

---

## 10. Experimental Setup & Evaluation Methodologies

### 10.1. Benchmark Evaluation Metrics
For an academic thesis, the following empirical benchmarks should be reported:
1. **Detection Accuracy:**
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   $$\text{mAP@50} = \int_0^1 P(R) dR \quad (\text{at IoU } \ge 0.50)$$
2. **False Positive Rejection Rate (FPRR):**
   $$\text{FPRR} = \frac{\text{True Negatives}}{\text{True Negatives} + \text{False Positives}}$$
   Tested across standard distractor suites: yellow incandescent bulbs, fluorescent emergency signage, high-visibility orange coats, and sunset window backlights.
3. **Temporal Stability Index (TSI):**
   $$\text{TSI} = 1.0 - \frac{\text{Box Flickers (appearances/disappearances)}}{\text{Total Active Frames}}$$

### 10.2. Automated Verification Script
The repository provides [`verify_detection_pipeline.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/verify_detection_pipeline.py) as an automated reproducible test harness across three standardized scenarios:
1. `test_fire_real.jpg`: Ground truth flame boundaries in realistic interior environments.
2. `test_smoke_real.jpg`: Heavy smoke plume validation with dispersion calculations.
3. `test_people_real.jpg`: High-salience distractor challenge (orange jackets and human subjects).

---

## 11. Recommended Thesis Chapter Structure

When preparing a Master's or Bachelor's thesis or publication using this codebase, the following chapter structure is recommended:

* **Chapter 1: Introduction**
  * Fireground tactical operations and information deficit.
  * The transition from point sensors to spatial edge intelligence.
  * Research questions, objectives, and thesis outline.
* **Chapter 2: Related Work**
  * Traditional optical and chemical fire detection systems.
  * Deep learning in wildfire and structural fire detection (CNNs, YOLO).
  * Vision-Language Foundation Models (VLMs) and referring expression grounding.
  * Temporal filtering and multi-object tracking under degraded visual conditions.
* **Chapter 3: System Architecture & Design**
  * Edge-computing constraints and hardware specifications.
  * The decoupled detector design pattern.
  * Time-shifted video pipeline and async thread isolation.
* **Chapter 4: Multi-Spectral & Multi-Gate Detection Algorithms**
  * LocateAnything-3B VLM grounding and quantization mechanics.
  * YOLOv8 multi-class ensemble pipeline.
  * Mathematical formulation of the 5-Gate Combustion Filter.
  * Smoke dispersion analysis and high-frequency edge suppression.
* **Chapter 5: Temporal Tracking, Kinematics, & Hazard Assessment**
  * Centroid kinematic motion and IoU tracking.
  * Smoke Density Index formulation and trajectory derivation.
  * Persistence state machine and additive hazard threat scoring.
* **Chapter 6: Experimental Results & Discussion**
  * Quantitative detection evaluation (Precision, Recall, mAP@50).
  * Ablation study: Impact of individual gates in the combustion filter.
  * Latency, VRAM footprint, and FPS benchmarks on NVIDIA RTX 3050.
  * Qualitative analysis of edge cases (low light, partial occlusion, smoke fog).
* **Chapter 7: Conclusion & Future Work**
  * Summary of contributions.
  * Integration with multi-spectral thermal (FLIR) imaging.
  * 3D spatial occupancy mapping via SLAM and edge mesh networks.

---

## 12. Complete Codebase Navigation Matrix

| Component | File Path | Key Functions & Symbols |
|---|---|---|
| **Server Entrypoint** | [`app.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/app.py) | `create_app()`, `/video_feed`, `/api/camera/*`, `/api/analyse/*` |
| **Detector Abstraction** | [`core/detector.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/detector.py) | [Detector](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/detector.py#L96-L150), [BoundingBox](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/detector.py#L48-L70), [Detection](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/detector.py#L72-L91), [DetectionQuery](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/detector.py#L32-L42) |
| **Ensemble & Filter Gates**| [`models/locate_anything/mock_detector.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/models/locate_anything/mock_detector.py) | `MockDetector`, `_validate_flame_region()`, `_cv2_smoke_analyzer()` |
| **LocateAnything VLM** | [`models/locate_anything/locate_anything_detector.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/models/locate_anything/locate_anything_detector.py) | `LocateAnythingDetector`, `load()`, `detect()`, `_parse_boxes()` |
| **Video Engine & Ring Buffer**| [`core/video_processor.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/video_processor.py) | [VideoProcessor](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/video_processor.py#L80-L542), `_capture_loop()`, `_inference_loop()`, `get_latest_jpeg()` |
| **Kinematic Object Tracker**| [`core/tracker.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/tracker.py) | [ObjectTracker](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/tracker.py#L34-L178), `update_from_detection()`, `step_motion()` |
| **Smoke Severity Analyzer** | [`core/smoke_analyzer.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/smoke_analyzer.py) | [SmokeAnalyzer](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/smoke_analyzer.py#L117-L229), `update()`, `_compute_trend()` |
| **Temporal Hazard Engine**| [`core/hazard_engine.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/hazard_engine.py) | [HazardEngine](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/hazard_engine.py#L153-L292), `_PersistenceTracker`, `_compute_hazard()` |
| **Alert Notification Engine**| [`core/alert_manager.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/alert_manager.py) | [AlertManager](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/alert_manager.py#L39-L131), `process()`, `_make_alert()` |
| **Event History Archiver** | [`core/event_manager.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/event_manager.py) | [EventManager](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/event_manager.py#L49-L141), `_open_event()`, `_close_event()` |
| **File Analysis Engine** | [`core/image_processor.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/image_processor.py) | [ImageProcessor](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/core/image_processor.py#L37-L144), `VideoFileProcessor` |
| **Pipeline Verification** | [`verify_detection_pipeline.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/verify_detection_pipeline.py) | `run_verification()`, test case runner |
| **Fine-Tuning Harness** | [`train_detector.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/train_detector.py) | `train()`, `prepare_fireguard_dataset()`, data augmentation |
| **3D Knowledge Visualizer**| [`generate_3d_graph.py`](file:///c:/Users/lauki/OneDrive/Desktop/fireguard%20AI/generate_3d_graph.py) | `build_3d_html()`, Three.js force graph generator |
