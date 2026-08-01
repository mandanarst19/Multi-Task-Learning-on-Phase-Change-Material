# Multi-Task Learning on Phase-Change Material Reservoir Computing

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Kaggle T4](https://img.shields.io/badge/platform-Kaggle%20T4-orange.svg)](https://kaggle.com)

A single fixed reservoir of 784 VO₂ thermal neuristors simultaneously solves three independent classification tasks — digit recognition, color classification, and parity detection — on a custom Colored MNIST dataset, with zero catastrophic forgetting. To our knowledge, this is the first demonstration of multi-task learning on a phase-change material reservoir.

---

## Results

| Task | Accuracy | Chance |
|------|----------|--------|
| Digit recognition (0–9) | **84.46%** | 10% |
| Color classification (10 colors) | **77.96%** | 10% |
| Parity detection (even/odd) | **87.66%** | 50% |

All three tasks share the same reservoir and the same forward pass. Zero catastrophic forgetting is guaranteed by construction — the reservoir is fixed and each readout is an independent linear layer, so tasks cannot interfere with each other. Task independence is verified statistically: χ²(10×10) = 95.27, p = 0.133 for digit–color predictions.

---

## Key Findings

**1 — Multi-task learning on phase-change material**  
Three simultaneous tasks on a single VO₂ reservoir with zero catastrophic forgetting by architectural design.

**2 — Noise-induced phase transition**  
Accuracy is stable across σ ∈ [0, 0.0002], then undergoes an abrupt transition between σ = 0.0002 and σ = 0.0005. Feature variance collapses ~18,000-fold. Color drops 50 pp while parity drops only 0.5 pp. There is no accuracy peak above the zero-noise baseline, ruling out Stochastic Resonance.

**3 — Timescale-task specialization**  
Color classification depends on the slow insulating timescale τ_ins ≈ 7.57 μs — accuracy drops 41 pp when restricted to the first 2 μs of the trajectory. Parity is stable across all temporal windows, consistent with τ_met ≈ 187 ns being sufficient. Confirmed independently by temporal masking and Ridge weight analysis.

**4 — Causal isolation of τ_ins**  
Scaling R_ins by r ∈ {1.0, 0.5, 0.2, 0.1} reduces τ_ins from 7.54 μs to 0.75 μs while leaving τ_met and τ_th unchanged. Color drops 33 pp; parity drops less than 1 pp. A simplified 2-class color task shows the same collapse, ruling out task difficulty as an explanation. Late temporal bins collapse while early bins remain stable, confirming τ_ins as the mechanism.

---

## Architecture

```
Input image (28×28 RGB)
        │
        ▼  luminance = 0.299R + 0.587G + 0.114B
  Voltage mapping  [10.5 V → 12.2 V]
        │
        ▼
┌───────────────────────────────────────┐
│  VO₂ Reservoir  (fixed, not trained)  │
│  784 thermal neuristors, 28×28 grid   │
│  thermally coupled, t_max = 10 μs     │
└───────────────────────────────────────┘
        │  current trajectory I(t)
        ▼
  Temporal max-pooling
  20 bins × 500 ns → 15,680-dim feature vector
        │
   ┌────┴──────────┬──────────────┐
   ▼               ▼              ▼
Readout_digit  Readout_color  Readout_parity
Ridge (α=1e-3) Ridge (α=1e-3) Ridge (α=1e-3)
```

---

## Repository Structure

```
├── experiments/
│   ├── 01_main_multitask.py     # main results and catastrophic forgetting verification
│   ├── 02_noise_sweep.py        # noise-induced phase transition
│   ├── 03_timescale_masking.py  # temporal window masking (Method 1)
│   ├── 04_weight_importance.py  # Ridge weight analysis (Method 2)
│   └── 05_causal_tau_ins.py     # causal isolation of τ_ins
├── docs/
│   └── physics_background.md
├── results/
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Reproducing Results

This code runs on Kaggle with a free T4 GPU. You will need to add the Zhang et al. simulator as a dataset (see Dependencies).

Upload the scripts from `experiments/` to `/kaggle/working/`, then run them in order:

```python
# Step 1 — Main results (~90 min). Also caches features for steps 3 and 4.
exec(open('/kaggle/working/01_main_multitask.py').read())

# Step 2 — Noise sweep (runs in seconds, no simulation needed)
exec(open('/kaggle/working/02_noise_sweep.py').read())

# Step 3 — Timescale masking (~10 min, loads cached features from step 1)
exec(open('/kaggle/working/03_timescale_masking.py').read())

# Step 4 — Weight importance (~2 min, loads cached features from step 1)
exec(open('/kaggle/working/04_weight_importance.py').read())

# Step 5 — Causal experiment (~6 hours total, crash-safe — re-run to resume)
exec(open('/kaggle/working/05_causal_tau_ins.py').read())
```

---

## Physical Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Array size | 28×28 = 784 neuristors | Zhang et al. |
| τ_met | ~187 ns | Zhang et al. |
| τ_th | ~241 ns | Zhang et al. |
| τ_ins | ~7.57 μs | Zhang et al. |
| T_c | 332.8 K | Zhang et al. |
| Noise strength σ | 0.0002 μJ·s⁻¹/² | this work |
| Temporal bins | 20 × 500 ns | this work |
| Feature dimension | 784 × 20 = 15,680 | this work |
| Input voltage | 10.5 – 12.2 V | this work |
| Ridge α | 1×10⁻³ | this work |

---

## Colored MNIST Dataset

MNIST images are colored with 10 distinct colors. The reservoir receives a single luminance channel computed as:

```
L = 0.299R + 0.587G + 0.114B  →  V = 10.5 + 1.7 × L
```

| Color | Label | RGB |
|-------|-------|-----|
| Red | 0 | (1.0, 0.0, 0.0) |
| Green | 1 | (0.0, 1.0, 0.0) |
| Blue | 2 | (0.0, 0.0, 1.0) |
| Yellow | 3 | (1.0, 1.0, 0.0) |
| Magenta | 4 | (1.0, 0.0, 1.0) |
| Cyan | 5 | (0.0, 1.0, 1.0) |
| Orange | 6 | (1.0, 0.5, 0.0) |
| Purple | 7 | (0.5, 0.0, 1.0) |
| Dark Green | 8 | (0.0, 0.5, 0.0) |
| Gray | 9 | (0.5, 0.5, 0.5) |

Color assignments are drawn uniformly at random, independent of digit identity. Independence is verified with a chi-square test on each split.

---

## Dependencies

The VO₂ simulator (Circuit2D) is from:

> Zhang, Y. et al. (2023). *Collective dynamics and long-range order in thermal neuristor networks.* arXiv:2312.12899v3.

All scripts locate the simulator automatically:

```python
for root, dirs, files in os.walk('/kaggle/input'):
    if 'model.py' in files:
        sys.path.insert(0, root)
        break
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contact

Mandana Roosta  
Master's Student in Physics, Shahid Beheshti University  
mandanaroosta.academia@gmail.com  
github.com/mandanarst19
