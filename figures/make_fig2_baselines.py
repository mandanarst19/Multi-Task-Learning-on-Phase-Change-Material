"""Fig. 2 — reservoir vs. baselines on the three tasks (same dataset, same P5 ridge readout, 60k).

Input : paper1_data.json  (key 'baselines_60k', incl. reservoir_sigma2e-4_per_seed)
Output: fig2_baselines.pdf / .svg / .png
Usage : python make_fig2_baselines.py paper1_data.json OUTDIR
"""
import sys, json, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
MM = 1 / 25.4
TASK_COL = {'digit': '#0072B2', 'color': '#D55E00', 'parity': '#009E73'}
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import iop_style as IOP
IOP.apply()

B = json.load(open(SRC))['baselines_60k']
PS = B['reservoir_sigma2e-4_per_seed']
for t in ('digit', 'color', 'parity'):                                  # per-seed values reproduce the plotted mean
    m = np.mean([PS[f'seed{k}'][t] for k in range(3)])
    assert abs(m - B['reservoir_sigma2e-4_3seeds_mean'][t]) < 6e-5, t
    assert abs(100 * np.std([PS[f'seed{k}'][t] for k in range(3)], ddof=1) - PS['std_ddof1_pp'][t]) < 6e-3, t
MODELS = [  # (json key, tick label) — dimensions only where the JSON records them
    ('A_raw_input_linear',              'Raw\ninput'),
    ('B_random_tanh_features',          'Random\ntanh'),
    ('D_per_pixel_sigmoids',            'Per-pixel\nsigmoid'),
    ('reservoir_sigma2e-4_3seeds_mean', 'Reservoir\nσ=2×10$^{-4}$'),
    ('reservoir_sigma0',                'Reservoir\nσ=0'),
]
TASKS = [('digit', 'Digit'), ('color', 'Colour'), ('parity', 'Parity')]

fig, ax = plt.subplots(figsize=(85 * MM, 58 * MM))
x = np.arange(len(MODELS)); w = 0.26
for j, (t, lab) in enumerate(TASKS):
    vals = [100 * B[k][t] for k, _ in MODELS]
    ax.bar(x + (j - 1) * w, vals, w, color=TASK_COL[t], edgecolor='white', lw=0.4, label=lab,
           hatch=IOP.TASK_HATCH[t])
    # error bar: sample s.d. of three noise seeds (only the noisy reservoir has seeds; A/B/D and σ = 0 are deterministic)
    i = [k for k, _ in MODELS].index('reservoir_sigma2e-4_3seeds_mean')
    ax.errorbar(x[i] + (j - 1) * w, vals[i], yerr=PS['std_ddof1_pp'][t], fmt='none', ecolor='k',
                elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=5)
ax.axvspan(len(MODELS) - 2.5, len(MODELS) - 0.5, color='0.94', zorder=0, lw=0)  # reservoir group
ax.set_xticks(x); ax.set_xticklabels([l for _, l in MODELS], linespacing=1.05)
ax.set_xlim(-0.5, len(MODELS) - 0.5)
ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel('Test accuracy (%)')
ax.yaxis.grid(True, color='0.88', lw=0.4); ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False,
          handlelength=1.2, columnspacing=1.4)

IOP.save(fig, f'{OUT}/fig2_baselines', IOP.SINGLE_MM)

print(f'{"model":34s} digit  colour parity  min(task)')
for k, l in MODELS:
    v = [100 * B[k][t] for t, _ in TASKS]
    print(f'{k:34s} {v[0]:6.2f} {v[1]:6.2f} {v[2]:6.2f}  {min(v):6.2f}')
