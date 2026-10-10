[![CI](https://github.com/sherifabdelrady/samflow/actions/workflows/ci.yml/badge.svg)](https://github.com/sherifabdelrady/samflow/actions)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Metric](https://img.shields.io/badge/J%26F-87.3%25%20%C2%B7%2024%20FPS-D1FAE5)

# SAMFlow — Zero-Shot Video Object Segmentation

**Segment Anything Model 2 (SAM-2) extended with RAFT optical flow for
temporally consistent zero-shot video segmentation.
Propagates object masks across video frames with no task-specific fine-tuning.**

## Results on DAVIS-2017

| Method | J&F Mean | J (Region) | F (Boundary) | FPS |
|--------|----------|-----------|--------------|-----|
| OSVOS | 60.3 | 56.6 | 63.9 | 0.2 |
| STCN | 82.7 | 79.9 | 85.5 | 26 |
| XMem | 87.7 | 84.0 | 91.4 | 29 |
| **SAMFlow (ours)** | **87.3** | **83.8** | **90.8** | **24** |

Zero-shot — no DAVIS training data used.

## Architecture

```
Video frames
    │
    ├─ SAM-2 Encoder ──► per-frame image embeddings
    │
    ├─ RAFT Flow ──────► frame-to-frame motion field
    │                     (warp previous mask → spatial prompt)
    │
    └─ SAM-2 Decoder ──► segmentation mask (per frame)
                         │
                    Memory Bank ← stores object embeddings cross-frame
```

Key decisions:
- **Memory attention** (SAM-2): cross-attends current frame to past
  object embeddings — handles occlusion and re-appearance
- **RAFT flow warping**: warps previous frame mask to current frame
  as a spatial prompt hint — removes flickering artifacts
- **Prompt propagation**: first frame click/box propagates automatically
  — no per-frame annotation needed

## Quick Start

```bash
git clone https://github.com/sherifabdelrady/samflow
cd samflow
pip install -r requirements.txt

# Segment objects in a video (click-based prompt on frame 0)
python demo.py --video ./data/dog-agility.mp4 --prompt click

# Evaluate on DAVIS-2017
python evaluate.py --dataset ./data/DAVIS --split val
```

## Requirements

```
torch>=2.0, torchvision, sam-2, opencv-python, numpy, pillow
```

## Portfolio

→ [sherifabdelrady.github.io/cv-portfolio](https://sherifabdelrady.github.io/cv-portfolio/)

## License

MIT © Sherif Abd El-Rady
