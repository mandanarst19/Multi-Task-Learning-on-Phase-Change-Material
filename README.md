# Multi-Task Learning on Phase-Change Material Reservoir Computing

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Kaggle T4](https://img.shields.io/badge/platform-Kaggle%20T4-orange.svg)](https://kaggle.com)

**First demonstration of multi-task learning on a phase-change material reservoir.**  
A single fixed reservoir of 784 VO₂ thermal neuristors simultaneously solves three independent classification tasks — digit recognition, color classification, and parity detection — on a custom Colored MNIST dataset, with zero catastrophic forgetting.

---

## Results

| Task | Accuracy | Random Baseline |
|------|----------|-----------------|
| Digit recognition (0–9) | **84.46%** | 10% |
| Color classification (10 colors) | **77.96%** | 10% |
| Parity detection (even/odd) | **87.66%** | 50% |

- All three tasks trained on the **same reservoir**, same forward pass, independent linear readouts
- **Zero catastrophic forgetting** — verified across all training orders (architectural guarantee)
- **Task independence** verified: χ²(10×10) = 95.27, p = 0.133 (digit–color)
- Training: single-pass Ridge Regression, ~90 minutes on Kaggle T4

---

## Key Findings

**1 — Multi-task learning on phase-change material**  
Three simultaneous tasks on a single VO₂ reservoir. Fixed reservoir + independent Ridge readouts guarantee zero catastrophic forgetting by construction.

**2 — Noise-induced phase transition**  
Performance stable across σ ∈ [0, 0.0002], then abrupt transition between σ = 0.0002 and σ = 0.0005:
- Feature variance collapses ~18,000-fold (~4.3 orders of magnitude)
- Color drops 50 pp, parity drops only 0.5 pp
- No peak above zero-noise baseline — this is a phase transition, not Stochastic Resonance

**3 — Timescale-task specialization**  
Color classification requires slow dynamics (τ_ins ≈ 7.57 μs):
- Accuracy drops 41 pp when restricted to early bins (0–2 μs)
- Parity stable across all temporal windows (τ_met ≈ 187 ns sufficient)
- Confirmed by two independent methods (temporal masking + weight importance)

**4 — Causal isolation of τ_ins**  
R_ins scaled by r ∈ {1.0, 0.5, 0.2, 0.1} reduces τ_ins from 7.54 μs to 0.75 μs while leaving τ_met and τ_th unchanged:
- Color drops 33 pp; parity drops < 1 pp
- 2-class color also collapses → task difficulty ruled out
- Late bins [16–19] collapse; early bins [0–3] remain stable → mechanism confirmed

---

## Architecture

```
Input image (28×28 RGB)
        │
        ▼ luminance: 0.299R + 0.587G + 0.114B
  Voltage Mapping
  [10.5 V → 12.2 V]
        │
        ▼
┌─────────────────────────────────────┐
│   VO₂ Reservoir  (fixed — no       │
│   training)                         │
│   784 thermal neuristors, 28×28     │
│   thermally coupled, t_max = 10 μs  │
└─────────────────────────────────────┘
        │  current trajectory I(t)
        ▼
  Temporal Max-Pooling
  20 bins × 500 ns → flatten
  → 15,680-dim feature vector
        │
   ┌────┴──────────┬──────────────┐
   ▼               ▼              ▼
Readout_digit  Readout_color  Readout_parity
 Ridge (α=1e-3)  Ridge (α=1e-3)  Ridge (α=1e-3)
   │               │              │
Digit (0–9)   Color (10 cls)  Parity (E/O)
```

---

## Repository Structure

```
├── experiments/
│   ├── 01_main_multitask.py       # Main results + catastrophic forgetting test
│   ├── 03_timescale_masking.py    # Method 1: temporal window masking
│   ├── 04_weight_importance.py    # Method 2: Ridge weight analysis
│   ├── 05a_r1p0.py                # Causal: r_scale=1.0, τ_ins=7.54 μs
│   ├── 05b_r05.py                 # Causal: r_scale=0.5, τ_ins=3.77 μs
│   ├── 05c_r02.py                 # Causal: r_scale=0.2, τ_ins=1.51 μs
│   ├── 05d_r01.py                 # Causal: r_scale=0.1, τ_ins=0.75 μs
│   ├── 05e_analysis.py            # Causal analysis + experiments A & B
│   └── 06_noise_sweep.py          # Noise-induced phase transition
├── docs/
│   └── physics_background.md      # VO₂ physics derivation
├── results/                        # JSON outputs from all experiments
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Reproducing Results

### Requirements

```bash
# Clone this repository
git clone https://github.com/mandanarst19/Multi-Task-Learning-on-Phase-Change-Material

# Install dependencies
pip install -r requirements.txt
```

This code requires the VO₂ neuristor simulator from Zhang et al. (2023).  
Add their dataset to your Kaggle notebook (see Dependencies below).

### Running on Kaggle (recommended — free T4 GPU)

Upload all scripts from `experiments/` to `/kaggle/working/`, then:

```python
# Experiment 01 — Main results (~90 min, saves features for 03 & 04)
exec(open('/kaggle/working/01_main_multitask.py').read())

# Experiment 03 — Timescale masking (~10 min, no simulation)
exec(open('/kaggle/working/03_timescale_masking.py').read())

# Experiment 04 — Weight importance (~2 min, no simulation)
exec(open('/kaggle/working/04_weight_importance.py').read())

# Experiment 05 — Causal τ_ins (one script per Kaggle session, ~90 min each)
exec(open('/kaggle/working/05a_r1p0.py').read())    # Session 1
exec(open('/kaggle/working/05b_r05.py').read())     # Session 2
exec(open('/kaggle/working/05c_r02.py').read())     # Session 3
exec(open('/kaggle/working/05d_r01.py').read())     # Session 4
exec(open('/kaggle/working/05e_analysis.py').read()) # Session 5 (~5 min)

# Experiment 06 — Noise sweep (~2 sec, results pre-verified)
exec(open('/kaggle/working/06_noise_sweep.py').read())
```

Features are cached after first extraction — restarted sessions load from disk automatically.

---

## Physical Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Array size | 28×28 = 784 neuristors | Zhang et al. |
| τ_met (metallic RC time) | ~187 ns | Zhang et al. |
| τ_th (thermal RC time) | ~241 ns | Zhang et al. |
| τ_ins (insulating RC time) | ~7.57 μs | Zhang et al. |
| T_c (transition temperature) | 332.8 K | Zhang et al. |
| Noise strength (optimal) | σ = 0.0002 μJ·s⁻¹/² | This work |
| Temporal bins | 20 × 500 ns | This work |
| Feature dimension | 784 × 20 = 15,680 | This work |
| Input voltage range | 10.5 – 12.2 V | This work |
| Ridge regularization α | 1×10⁻³ | This work |

---

## Colored MNIST Dataset

Standard MNIST images colored with 10 distinct colors using ITU-R BT.601 luminance conversion:

```
Gray = 0.299R + 0.587G + 0.114B → Voltage ∈ [10.5, 12.2] V
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

Color assignments are random and independent of digit identity — verified with chi-square test.

---

## Dependencies

The VO₂ neuristor simulator (Circuit2D) is from:

> Zhang, Y. et al. (2023). *Collective dynamics and long-range order in thermal neuristor networks.* arXiv:2312.12899v3.

All scripts auto-detect the simulator path:

```python
for root, dirs, files in os.walk('/kaggle/input'):
    if 'model.py' in files:
        sys.path.insert(0, root)
        break
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Contact

Mandana Roosta 
Master's Student in Physics  
Shahid Beheshti University  
📧 mandanaroosta.academia@gmail.com  
🔗 [github.com/mandanarst19](https://github.com/mandanarst19)

