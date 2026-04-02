# Project Proposal: Household Appliance State Detection for Energy-Aware Smart Home Assistance

**Course**: Computer Vision / Embodied AI
**Author**: Kailai Li

---

## Title

Household Appliance State Detection from RGB Images for Energy-Aware Smart Home Assistance

---

## Problem Statement

Smart home systems often rely on dedicated hardware sensors (motion detectors, smart plugs, contact sensors) to monitor appliance usage. These solutions are costly to deploy and lack generalization across different home layouts.

This project proposes a **vision-based** alternative: using a standard RGB camera and a lightweight two-stage perception pipeline to detect common household objects and infer their operational states (e.g., light on/off, door open/closed, TV active/inactive). The system then applies a simple rule-based advisor to suggest energy-saving actions based on detected states and contextual cues (e.g., time of day, room occupancy).

**Research Question**:

> Can a lightweight two-stage vision pipeline — object detection followed by binary state classification — reliably infer the operational state of common household appliances from a single RGB image?

---

## Motivation

- **Practical value**: Unnecessary appliance usage (lights left on, doors left open) accounts for a measurable fraction of household energy waste. A camera-based system requires no additional sensor hardware.
- **Well-scoped for a course project**: The task decomposes cleanly into detection and classification, both of which have mature tooling and pretrained baselines.
- **Non-trivial challenges**: Real indoor scenes involve clutter, partial occlusion, varying lighting conditions, and viewpoint changes — these make the binary state classification problem genuinely interesting beyond running a pretrained detector off-the-shelf.
- **Evaluation is concrete**: Accuracy, F1-score, and confusion matrices per appliance category are straightforward to compute on a labeled test set.

---

## Scope and Simplifications

| Original Idea | Simplified Version | Reason |
|---|---|---|
| Embodied agent moving through 3D space | Fixed-camera single RGB image input | 3D reconstruction requires depth sensors + SLAM; out of scope |
| Full scene understanding | Binary state per object instance | Tractable classification task with clear ground truth |
| Joint recognition + energy behavior reasoning | Detect → classify → rule-based advisor | Clean modular pipeline; each stage independently testable |
| Novel model architecture | Pretrained YOLOv8 + lightweight classifier | Focus on the state classification problem, not detection from scratch |

---

## System Pipeline

```
┌─────────────────────────────────────────┐
│          Input: RGB Image               │
│  (single frame from fixed camera)       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│     Stage 1: Object Detection           │
│  YOLOv8 (pretrained on COCO)            │
│  → bounding boxes + class labels        │
│  Target classes: lamp, door, TV,        │
│  refrigerator, microwave, oven, fan     │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│     Stage 2: State Classification       │
│  For each detected object crop:         │
│  → MobileNetV3 / ResNet-18 classifier   │
│  → Binary label: ON / OFF               │
│     (or OPEN / CLOSED for doors/fridges)│
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│     Stage 3: Rule-Based Energy Advisor  │
│  Input: detected states + time of day   │
│  Rules (examples):                      │
│  - Light ON + no person detected → alert│
│  - Fridge door OPEN > threshold → alert │
│  - TV ON + no person detected → alert   │
│  Output: natural language suggestion    │
└─────────────────────────────────────────┘
```

---

## Target Object Categories and States

| Object | State Labels | Detection Challenge |
|--------|-------------|---------------------|
| Ceiling / desk lamp | ON / OFF | Brightness of light region |
| Door | OPEN / CLOSED | Angle, gap visibility |
| Television | ON / OFF | Screen brightness + content |
| Refrigerator door | OPEN / CLOSED | Door edge angle |
| Microwave | ON / OFF | Display / interior light |
| Electric fan | ON / OFF | Blade blur (motion blur cue) |

---

## Dataset

### Option A — Self-Collected Small Dataset (Recommended for Phase 1)
- Collect 150–250 images per object category (ON and OFF balanced)
- Sources: own home, public indoor image repositories, video frame extraction
- Annotate with bounding boxes (LabelImg / Roboflow) and binary state labels
- Train/Val/Test split: 70 / 15 / 15

### Option B — Public Datasets (Supplement or Replace)
| Dataset | Relevance |
|---------|-----------|
| **OpenImages v7** | Contains many household objects with bounding box annotations |
| **COCO 2017** | Baseline detection; lacks state labels but useful for pretraining |
| **HACS / ActivityNet** | Video-level; can extract frames for appliance state mining |
| **SmartHome (CMU / TU Berlin)** | Activity recognition in home environments |

**Phase 1**: Use self-collected images for state classification.
**Phase 2** (optional): Scale up with OpenImages crops + manual state annotation.

---

## Models and Tools

| Component | Tool / Model | Justification |
|-----------|-------------|---------------|
| Object detection | **YOLOv8n** (Ultralytics) | Pretrained on COCO, fast inference, good on household objects |
| State classifier | **MobileNetV3-Small** fine-tuned | Lightweight; suitable for edge deployment; easy to fine-tune |
| Data annotation | **Roboflow** or **LabelImg** | Free, fast bounding box + label annotation |
| Training framework | **PyTorch + torchvision** | Standard, well-documented |
| Experiment tracking | **Weights & Biases** (optional) | Track fine-tuning runs across categories |

---

## Evaluation Metrics

### Detection (Stage 1)
- mAP@0.5 on target household object categories
- Baseline: YOLOv8n zero-shot on COCO classes

### State Classification (Stage 2)
- Per-class Accuracy, Precision, Recall, F1-score
- Confusion matrix per object category
- Robustness evaluation: performance under occlusion / low light / unusual viewpoint

### End-to-End (Stage 3)
- Advisor correctness rate: fraction of frames where the energy suggestion is appropriate
- False alert rate: advisor triggers when no action is needed

---

## Experiment Plan

### Experiment 1 — Baseline State Classification
- Train MobileNetV3 on cropped object images (lamp, door, TV)
- Report per-category F1 on held-out test set
- Compare: zero-shot CLIP vs fine-tuned MobileNetV3

### Experiment 2 — Effect of Input Crop Quality
- Compare state classification accuracy using:
  - Ground-truth bounding box crops
  - YOLOv8 predicted crops (realistic pipeline)
- Quantify the detection-to-classification error propagation

### Experiment 3 — Robustness Ablation
- Evaluate classifier on images with:
  - Partial occlusion (manually masked 20–40% of object)
  - Low light (brightness reduced in post-processing)
  - Off-angle viewpoint (rotated crops)
- Report F1 degradation per condition

### Experiment 4 — Rule-Based Advisor Evaluation
- Construct 30–50 annotated test scenarios with known ground-truth actions
- Measure advisor precision and recall against human-labeled "correct action"

---

## Development Roadmap

```
Week 1–2:  Data collection and annotation
           - Collect 100-200 images per target category
           - Annotate with Roboflow (bounding box + state label)
           - Set up PyTorch data loaders

Week 3–4:  Stage 1 + Stage 2 implementation
           - Run YOLOv8 zero-shot baseline; evaluate on target classes
           - Fine-tune MobileNetV3 on annotated state crops
           - Experiment 1: baseline classification results

Week 5–6:  Pipeline integration + robustness experiments
           - Connect detection output to classifier input
           - Experiment 2: crop quality effect
           - Experiment 3: robustness ablation

Week 7–8:  Energy advisor + final evaluation
           - Implement rule-based advisor (Stage 3)
           - Experiment 4: end-to-end advisor evaluation
           - Write report + assemble results tables and figures
```

---

## Limitations and Honest Scope

- **No temporal reasoning**: each image is analyzed independently; no tracking across frames
- **Rule-based advisor is not learned**: energy suggestions follow hand-crafted rules, not learned behavior
- **Small dataset**: state classification performance depends on annotation quality and diversity
- **Fixed camera assumption**: performance may degrade with unusual camera angles not seen in training

These are acceptable limitations for a course project. They also naturally define the boundary between this project and future work (e.g., video-based temporal state tracking, learned advisor policies).

---

## Connections to Broader Embodied AI

While this project uses a fixed camera (not a moving agent), the perception pipeline is a direct building block for embodied systems. A robot navigating a home could use the same detection + state classification modules to reason about which appliances need attention, informing its action planning. This grounds the project in the embodied AI setting without requiring robotics hardware.

---

## References

1. Redmon J, Farhadi A. YOLOv3: An incremental improvement. arXiv:1804.02767. 2018.
2. Jocher G et al. Ultralytics YOLOv8. GitHub. 2023.
3. Howard A et al. Searching for MobileNetV3. ICCV 2019.
4. Radford A et al. Learning transferable visual models from natural language supervision (CLIP). ICML 2021.
5. Deng J et al. ImageNet: A large-scale hierarchical image database. CVPR 2009.
6. Kuznetsova A et al. The Open Images Dataset V4. IJCV 2020.
