# Multi-Task-Learning-on-Phase-Change-Material
Exploiting VO₂ Phase-Transition Dynamics for Multi-Task Reservoir Computing Without Catastrophic Forgetting
> **First demonstration of multi-task learning using phase-change materials.**  
> A single fixed reservoir of 784 VO₂ thermal neuristors simultaneously solves three independent classification tasks — digit recognition, color classification, and parity detection — on a custom Colored MNIST dataset, with no catastrophic forgetting.

---

## Table of Contents
- [Abstract](#abstract)
- [Key Results](#key-results)
- [Architecture Overview](#architecture-overview)
- [Reproducing Results](#reproducing-results)
- [Physics Background](#physics-background)
- [Citation](#citation)
- [Contact](#contact)

---

## Abstract

This work presents the first demonstration of multi-task learning using VO₂ (vanadium dioxide) thermal neuristors as a physical reservoir computing substrate. A 28×28 array of thermally coupled VO₂ neuristors — each implementing a physical leaky integrate-and-fire neuron through coupled electrical and thermal dynamics — processes grayscale images and simultaneously trains three independent linear readout layers for digit recognition (0–9), color classification (Red/Green/Blue), and parity detection (Even/Odd), operating on a custom Colored MNIST dataset.

The reservoir exploits the insulator-to-metal phase transition of VO₂ at ~340 K, producing rich spatiotemporal spike patterns that are captured via a novel **temporal pooling** approach: continuous current trajectories are binned into 20 temporal windows of 500 ns each, yielding 15,680-dimensional feature vectors that encode both which neurons fired and when. Because the reservoir dynamics remain entirely fixed during training and only three independent linear readout layers are optimised via ridge regression, catastrophic forgetting is eliminated by construction. The system achieves **86.89% digit accuracy**, **76.50% color accuracy** (from grayscale luminance cues alone), and **88.73% parity accuracy**, with all tasks statistically independent (χ² = 63.51, p = 0.924).

---

## Key Results

| Task | Accuracy | Random Baseline | Method |
|------|----------|-----------------|--------|
| Digit Recognition (0–9) | **86.89%** | 10.0% | Ridge Regression |
| Color Classification (R/G/B) | **76.50%** | 33.3% | Ridge Regression |
| Parity Detection (Even/Odd) | **88.73%** | 50.0% | Ridge Regression |

**Additional highlights:**
- Training time: ~77 minutes (single pass, Ridge Regression) on Kaggle Tesla T4
- Feature dimensionality: 15,680 (784 neurons × 20 temporal bins)
- Task independence verified: χ²(4) = 63.51, p = 0.924
- Zero catastrophic forgetting (fixed reservoir, independent linear readouts)
- 32× more information density than binary spike-counting methods

---

## Architecture Overview

```
Input (28×28 MNIST pixel)
        │
        ▼
  Voltage Mapping
  [10.5 V → 12.2 V]
        │
        ▼
┌───────────────────────────────────┐
│   VO₂ Reservoir (fixed weights)   │
│   784 thermal neuristors          │
│   28×28 grid, thermally coupled   │
│   t_max = 10 µs,  dt = 10 ns     │
└───────────────────────────────────┘
        │  Current trajectory I(t)
        ▼
  Temporal Pooling
  500 ns bins → 20 time windows
  Flatten → 15,680-dim feature vector
        │
   ┌────┴────────┬────────────┐
   ▼             ▼            ▼
Readout_digit  Readout_color  Readout_parity
(15680→10)     (15680→3)      (15680→2)
Ridge Reg.     Ridge Reg.     Ridge Reg.
   │             │             │
Digit (0-9)   Color (R/G/B)  Parity (E/O)
```

---

```

### GPU Setup (recommended)

A CUDA-capable GPU with ≥ 8 GB VRAM is recommended. The code was developed and tested on an NVIDIA Tesla T4 (15 GB). CPU execution is supported but will be significantly slower for the reservoir simulation step.

---

**Expected console output:**
```
======================================================================
TEMPORAL POOLING SNN - GRAYSCALE ENCODING (PAPER'S APPROACH)
======================================================================
✓ GPU Optimized
✓ 500ns temporal windows | 20 temporal bins
✓ 784 neurons → 15,680 features
Extracting reservoir features: 100%|████████| 300/300 [72:40<00:00]

RESULTS:
  Digit:  86.89%
  Color:  76.50%
  Parity: 88.73%
  Training Duration: 4600.6s
======================================================================
```

### 2. Use the reservoir as a feature extractor

```python
import torch
from src.reservoir import Reservoir2D

# Initialise reservoir with paper's validated hyperparameters
reservoir = Reservoir2D(
    batch=50,
    Nx=28, Ny=28,
    N_out=10,
    V_min=10.5,     # V — maps to zero pixel intensity
    V_max=12.2,     # V — maps to max pixel intensity
    Cth_factor=0.15,
    noise_strength=0.2e-3
)
reservoir.eval()

# Extract features from a batch of images (values in [0, 1])
images = torch.rand(50, 784)  # 50 images, flattened 28×28
with torch.no_grad():
    features = reservoir.reservoir_func(images)  # → (50, 15680)

print(f"Feature shape: {features.shape}")  # torch.Size([50, 15680])
```

### 3. Load pre-trained classifiers and run inference

```python
import pickle
from tasks.temporal_pooling_rgb_comprehensive import extract_features_only

# Load trained classifiers
with open('results/ridge_classifiers.pkl', 'rb') as f:
    classifiers = pickle.load(f)

# Extract reservoir features for new images
features = extract_features_only(new_images, device='cuda')

# Predict all three tasks simultaneously
digit_preds  = classifiers['digit'].predict(features)
color_preds  = classifiers['color'].predict(features)
parity_preds = classifiers['parity'].predict(features)
```

---

## Reproducing Results

### On Kaggle (recommended — free T4 GPU)

1. Upload the repository to a Kaggle Dataset.
2. Open `notebooks/multi_task_demo.ipynb` in a Kaggle Notebook.
3. Enable GPU (Settings → Accelerator → GPU T4 × 1).
4. Run all cells. Total runtime: ~80 minutes.

### Key hyperparameters (from Zhang et al., arXiv:2312.12899v3)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `V_min` | 10.5 V | Input voltage for zero pixel intensity |
| `V_max` | 12.2 V | Input voltage for maximum pixel intensity |
| `Cth_factor` | 0.15 | Thermal capacitance scaling (Cth = 7.44 pJ/K) |
| `noise_strength` | 0.2 µJ·s⁻¹/² | Stochastic fluctuation amplitude |
| `t_max` | 10 µs | Simulation duration per image |
| `dt` | 10 ns | Integration time step (Euler–Maruyama) |
| `Δt_bin` | 500 ns | Temporal pooling window size |
| `n_bins` | 20 | Number of temporal bins |
| `R_load` | 12 kΩ | Load resistance |
| `T_base` | 325 K | Ambient temperature |
| `couple_factor` | 0.02 | Nearest-neighbour thermal coupling strength |
| `α_digit` | 1×10⁻³ | Ridge regularisation — digit task |
| `α_color` | 1×10⁻³ | Ridge regularisation — color task |
| `α_parity` | 1×10⁻⁴ | Ridge regularisation — parity task |

### Colored MNIST Dataset

Standard MNIST images are converted to a three-class color dataset by scaling pixel intensities:

| Color | Label | Luminance Factor | Approximate Grayscale Equivalent |
|-------|-------|-----------------|----------------------------------|
| Red   | 0     | 1.00            | Bright (full intensity) |
| Green | 1     | 0.70            | Medium intensity |
| Blue  | 2     | 0.50            | Dim intensity |

Color assignments are **random and independent of digit identity** (verified statistically). The reservoir receives a single grayscale channel computed as:

```python
# Voltage-mapped grayscale input to reservoir
V_input = V_min + (V_max - V_min) * (pixel_intensity / 255.0) * color_factor
```

---

## Physics Background

VO₂ is a correlated electron material that undergoes a sharp insulator-to-metal phase transition (IMT) near 340 K, with resistance dropping ~3 orders of magnitude. Each neuristor in the 28×28 array implements a physical leaky integrate-and-fire neuron:

**Electrical dynamics:**
$$C \frac{dV}{dt} = \frac{V_\text{in} - V}{R_\text{load}} - \frac{V}{R(T)}$$

**Thermal dynamics:**
$$C_\text{th} \frac{dT}{dt} = I^2 R(T) - S_\text{env}(T - T_0) + S_c \nabla^2 T + \sigma \xi(t)$$

**VO₂ resistance (hysteresis model):**
$$R(T, \delta) = R_m + \frac{R_0 e^{E_a/T} - R_m}{\left[1 + e^{-(T - T_c - w(1-\delta)/2)\,/\,(w/\beta)}\right]^\gamma}$$

The three distinct physical timescales — τ_metallic ≈ 187 ns, τ_thermal ≈ 241 ns, τ_insulating ≈ 7.57 µs — create a ~40× separation that enables complex temporal processing ideal for reservoir computing.

For a complete derivation and parameter justification, see `docs/physics_background.md`.

---

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{[PLACEHOLDER: citekey],
  author    = {Roosta, Mandana and [Mohseni, Majid]},
  title     = {Exploiting VO₂ Phase-Transition Dynamics for Multi-Task Reservoir Computing Without Catastrophic Forgetting},
  journal   = {[: IEEE TNNLS / journal name]},
  year      = {[: 2026]},
  volume    = {[: volume]},
  pages     = {[: pages]},
  doi       = {[: DOI]},
  url       = {[: URL]}
}
```

The underlying VO₂ neuristor physics model is based on:

```bibtex
@article{zhang2024collective,
  author    = {Zhang, [PLACEHOLDER: full author list]},
  title     = {Collective dynamics and long-range order in thermal neuristor networks},
  journal   = {Nature Commnucation },
  year      = {2024},

}
```

---

## License

This project is licensed under the MIT License — see (LICENSE) for details.

---

## Contact

**Mandana Roosta**  
Master's Student in Physics  
[Shahid Beheshti University /Physics Department]  
📧 [mandanaroosta.academia@gmail.com]  
🔗 [https://github.com/mandanarst19]


