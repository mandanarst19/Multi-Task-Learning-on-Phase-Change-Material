# Code and data: multi-task reservoir computing with a simulated VO₂ neuristor array (Paper 1)

This package holds every script and data file needed to reproduce the numbers and figures of the paper:
- a 28 × 28 array of thermally coupled VO₂ neuristors (device model of Zhang et al., *Nat. Commun.* **15**, 6986 (2024));
- Colored MNIST with three tasks: digit, colour and parity;
- a linear (ridge) readout and a logistic readout.

Package assembled 2026-11-03.

```
model/          model.py — VO₂ device and circuit model (Zhang et al. code, unchanged)
simulations/
  reconstructed/  R1–R6: scripts that replace notebooks lost after the original runs (verified, see below)
  standalone/     the original scripts that produced the remaining results
figures/        make_fig*.py + iop_style.py: regenerate every figure from data/
data/           paper1_data.json (every number in the paper, with its source) + simulation outputs used by the figures
```

## 1. Which script produces which result

| Result in the paper | Script | Output | Key in `data/paper1_data.json` |
|---|---|---|---|
| Reservoir, σ = 0, 60k: all time windows, sliding 1-µs windows, confusion matrices (Fig. 2, Fig. 3a–b, Fig. 4a) | `reconstructed/R1_final_sigma0_60k.py` | `result_sigma0_60k.json` | `sigma0_60k_final` |
| Reservoir, σ = 2×10⁻⁴, three noise seeds, 60k (Fig. 2, error bars) | `reconstructed/R2_noise_seeds_sigma2e-4.py` | `readout_s{0,1,2}.json` | `baselines_60k.reservoir_sigma2e-4_*` |
| Baselines A (raw input), B (random tanh features) and D (per-pixel sigmoids), 60k (Fig. 2) | `reconstructed/R3_baselines_ABD.py` | `baselines.json` | `baselines_60k` |
| σ = 0 switching simulation of 1,000 test images and the 10 Fig. 1 images (Fig. 1, Fig. 3c) | `reconstructed/R4_fig2_sim.py` | `fig2_sim.npz` | `switching_sigma0` |
| Switching and reversal diagnostics vs σ, C₀ and dt (Fig. 5d) | `reconstructed/R5_diagnostics_dt.py` | `diagnostics.json` | `diagnostics_dt` |
| First-switch jitter over 8 noise realisations (text) | `reconstructed/R6_jitter.py` | `jitter.json` | `jitter_sigma2e-4` |
| C₀ × σ grid, 20k (Fig. 5a–c) | `standalone/c0_sigma_grid_kaggle.py` | `c0sigma/*.json` | `c0_sigma_grid_20k` |
| Logistic readout on the σ = 0 features and on the raw input (Fig. 4b) | `standalone/logistic_sigma0_kaggle.py` | `logistic_sigma0_60k_run2.json` | `logistic_sigma0_60k` |
| Coupled vs uncoupled reservoir and current snapshots of image 78 (Fig. 3d, Fig. 6) | `standalone/coupling0_kaggle.py` | `coupling/*.json`, `snapshots_img78.npz` | `coupling_control_20k` |
| Coupling sweep c = 0–0.04 (Fig. 6a) | `standalone/coupling_sweep_kaggle.py` | `coupling_sweep/*.json`, `snapshots_img78_sweep.npz` | `coupling_sweep_20k` |
| Baseline D: ridge 20k and logistic readout (Fig. 4b, Fig. 6) | `standalone/logistic_D_kaggle.py` | `logistic_D.json` | `baseline_D_logistic_and_20k` |
| Baseline D: logistic readout, C extended to 1 and 10 (confirms C = 0.1 is an interior optimum) | `standalone/logistic_D_Cext_kaggle.py` | `logistic_D.json` (folder `logistic_D_Cext`) | `baseline_D_logistic_and_20k.logistic_60k_C_extension` |
| Uncoupled reservoir, 60k: ridge and logistic readout (Fig. 6) | `standalone/logistic_uncoupled_kaggle.py` | `logistic_uncoupled.json` | `uncoupled_60k_ridge_logistic` |
| Colour–label independence of the dataset (χ² test, Methods) | `standalone/color_independence_kaggle.py` | `color_independence.json` | `colour_label_independence` |
| R–T hysteresis of the device model (Fig. A1) | `standalone/rt_loop_kaggle.py` | `rt_loop.json` | `vo2_R_325K` |

Regenerating the figures (no simulation needed; run from this folder):

```
python figures/make_fig1.py         data/fig2_sim.npz OUT
python figures/make_fig2_baselines.py data/paper1_data.json OUT
python figures/make_fig3.py         data/paper1_data.json data/fig2_sim.npz data/snapshots_img78.npz OUT
python figures/make_fig4.py         data/paper1_data.json OUT
python figures/make_fig5.py         data/paper1_data.json OUT
python figures/make_fig6.py         data/paper1_data.json OUT
python figures/make_figA1.py        data/rt_loop.json OUT
```

## 2. Verification of the reconstructed scripts

The notebooks that produced the R1–R6 results were lost after the runs. The scripts R1–R6 were rewritten from the surviving code, which is the shared core of `c0_sigma_grid_kaggle.py`. Each one was then run in full and compared with the original results (2026-11-02/03, Kaggle, T4 GPU):

| Script | Type | Agreement with the original results |
|---|---|---|
| R1 | deterministic | all 27 accuracies (9 windows × 3 tasks) within 0.02 percentage points; same 263 masked features; digit and parity confusion matrices identical |
| R3 | deterministic | A and D identical; B differs by one test image (colour 43.19 vs 43.18 %) |
| R4 | deterministic | 783,993 of 784,000 first-switch times identical; 7 differ by one time step (10 ns) |
| R2 | stochastic | three-seed mean 87.35 / 78.33 / 89.99 % vs 87.37 / 78.19 / 90.00 %, within the seed-to-seed spread |
| R5 | stochastic | share of neurons with ≥ 3 switches, median switches and median reversals/µs identical in all 24 configurations |
| R6 | stochastic | first-switch jitter 54.9 vs 55.2 ns (C₀ × 1.0), identical for C₀ × 0.5 and σ = 0 |

Known differences (all within numerical reproducibility):
- **R1:** in the full window, a validation tie (1–2 of 10,000 images) selects a different colour α. As a result, colour accuracy is 81.47 % here and 81.48 % in the original run.
- **R2:** the individual noise realisations differ from the original ones; only the statistics agree.
- **R5:** the mean switch count is 0.03–0.2 % lower than in the original run. This quantity is not used in the paper.

Details: `data/paper1_data.json` → `reconstruction_checks`. The numbers in the paper are those of the original runs.

## 3. How to run

- **Environment.** Every simulation script is a standalone Kaggle notebook cell (Python 3; numpy, torch, torchvision; scikit-learn for the logistic scripts). The figure scripts need matplotlib, and `make_fig1.py` also needs schemdraw. Package versions are those of the Kaggle image at run time; they are not pinned. Accelerator: NVIDIA T4 GPU.
- **Inputs.** Each script needs `model/model.py` and `data/paper1_data.json`, which it uses for dataset checks and to print the reference values. It also needs Internet access to download MNIST. The scripts find the inputs under `/kaggle/input` or `/kaggle/working`. To run elsewhere, edit the `find()` function and the `/kaggle/working` paths.
- **Smoke test first.** Each simulation script starts with `SMOKE = True`, a small test run. Set `SMOKE = False` for the full run.
- **Approximate T4 runtimes (full run):** R1 ≈ 20 min, R2 ≈ 13 min per seed, R3 ≈ 11 min, R4 ≈ 10 min, R5 ≈ 70 min, R6 ≈ 15 min, logistic on the uncoupled reservoir ≈ 4–5 h. The other scripts give their runtime in their header.
- **Dataset.** Colours are drawn with `np.random.seed(42)`, then `randint(0, 10, 60000)` for the training set and `randint(0, 10, 10000)` for the test set. Input voltage: V = 10.5 + 1.7 · L_c · p V, where p is the pixel value and L_c the luminance of colour c. The scripts check this against the stored per-colour test counts.

## 4. Not included

- **P1 noise sweep.** This sweep used a fixed α = 10⁻³ and no feature standardisation. Its code is in the GitHub repository `multi-task-learning-on-phase-change-material` (`src/01`, `src/02`). It is needed only if the P1 results are reported.
- **`model/model.py`.** This file is the code of Zhang et al. Before releasing this package publicly, check the licence of their code release and cite it. If redistribution is not allowed, replace the file with a link to their repository.
