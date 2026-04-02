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

Scope reduced to **3 categories** for a 1-month timeline:

| Object | State Labels | Detection Challenge |
|--------|-------------|---------------------|
| Desk / ceiling lamp | ON / OFF | Brightness of light region |
| Door | OPEN / CLOSED | Angle, gap visibility |
| Television | ON / OFF | Screen brightness + content |

Refrigerator, microwave, and fan are dropped — they share the same approach and add collection overhead without new insights.

---

## Dataset

### Self-Collected Small Dataset
- **50–80 images per class per category** (ON and OFF balanced) = ~300–480 images total
- Sources: own home + Google Images / Unsplash for variety
- Collect over 2–3 days; annotate with **Roboflow** (free, browser-based, fast)
- Train/Val/Test split: 70 / 15 / 15

This size is intentionally small: the goal is to show that fine-tuning a pretrained model on very few examples still outperforms a zero-shot baseline, not to build a production-scale dataset.

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

| Stage | Metric | Notes |
|-------|--------|-------|
| Detection | Visual inspection + detection rate on 30 test images | YOLOv8 pretrained; no retraining needed |
| State classification | Accuracy, F1-score per category | Main result; compare CLIP vs fine-tuned |
| End-to-end pipeline | Accuracy drop (GT crops vs predicted crops) | Measures detection→classification error propagation |
| Energy advisor | Qualitative demo only | 5–10 sample outputs; no formal metric |

---

## Experiment Plan

Two core experiments fit within 1 month. The energy advisor serves as a qualitative demo, not a formal experiment.

### Experiment 1 — Baseline State Classification (Core)
- Fine-tune MobileNetV3 on ground-truth cropped images (lamp, door, TV)
- Compare against zero-shot CLIP as baseline
- Report per-category Accuracy and F1 on held-out test set
- **Deliverable**: table showing CLIP vs fine-tuned across 3 categories

### Experiment 2 — End-to-End Pipeline Evaluation (Core)
- Run full pipeline: YOLOv8 detection → MobileNetV3 classification on 30 new test images
- Compare accuracy using ground-truth crops vs YOLOv8-predicted crops
- Quantify how detection errors affect state classification
- **Deliverable**: accuracy drop table (GT crops vs predicted crops) + 5–10 qualitative examples

### Demo — Rule-Based Energy Advisor (Qualitative)
- Implement 3 simple rules (light ON → suggest off, door OPEN → alert, TV ON → suggest off)
- Show 5–10 sample outputs on test images as qualitative demo
- No formal evaluation; purely illustrative

---

## Development Roadmap (4 Weeks)

```
Week 1:  Data collection and annotation                    [~10 hrs]
         - Collect 50–80 images per class × 3 categories × 2 states
         - Annotate bounding boxes + state labels in Roboflow
         - Export dataset; set up PyTorch data loaders

Week 2:  Stage 1 + Stage 2 implementation                 [~10 hrs]
         - Run YOLOv8n zero-shot on test images; verify it detects
           lamp / door / TV reliably
         - Fine-tune MobileNetV3 on ground-truth crops
         - Experiment 1: CLIP vs fine-tuned results table

Week 3:  Pipeline integration + Experiment 2              [~8 hrs]
         - Connect YOLOv8 output crops → MobileNetV3 input
         - Run end-to-end on 30 test images
         - Compute accuracy with GT crops vs predicted crops
         - Implement 3-rule energy advisor; collect demo outputs

Week 4:  Analysis, visualization, and report              [~8 hrs]
         - Assemble results tables + qualitative examples
         - Write report (introduction, pipeline, experiments, conclusion)
         - Final cleanup and submission
```

**Total estimated effort**: ~36 hours over 4 weeks (~9 hrs/week)

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
