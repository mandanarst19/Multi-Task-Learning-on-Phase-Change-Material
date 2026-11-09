"""Fig. 5 — noise robustness and charging time (P5, 20k) and the dt dependence of the switching threshold.

Input : paper1_data.json  (c0_sigma_grid_20k, diagnostics_dt)
Output: fig5_noise.pdf / .svg / .png
Usage : python make_fig5.py paper1_data.json OUTDIR
"""
import sys, json, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

JS, OUT = sys.argv[1], sys.argv[2]
MM = 1 / 25.4
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import iop_style as IOP
IOP.apply()
# C0 levels: distinct non-task colours + markers (task colours are reserved for digit / colour / parity)
C0S = [(c, IOP.C0_COL[c], m) for c, m in ((0.5, 'o'), (1.0, 's'), (1.5, '^'))]
TASKS = [('digit', 'Digit'), ('color', 'Colour'), ('parity', 'Parity')]

J = json.load(open(JS)); R = J['c0_sigma_grid_20k']['runs']; DG = J['diagnostics_dt']['rows']
SIG = [(0.0, '0'), (2e-4, '2e-4'), (3e-4, '3e-4')]
run = lambda c0, s: R[f'c0_{c0}_sigma{s}']
xs = [1e4 * s for s, _ in SIG]

fig = plt.figure(figsize=(150 * MM, 120 * MM))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.95, wspace=0.5)
axA = [fig.add_subplot(gs[0, k]) for k in range(3)]
axB = fig.add_subplot(gs[1, 0]); axC = fig.add_subplot(gs[1, 1]); axD = fig.add_subplot(gs[1, 2])

# ---------------- (a) accuracy vs noise for three charging times ----------------
for ax, (t, lab) in zip(axA, TASKS):
    for c0, col, mk in C0S:
        ax.plot(xs, [100 * run(c0, s)['test_acc'][t] for _, s in SIG], '-' + mk, color=col, ms=3.5, lw=1.0,
                mec='k', mew=0.4)
    ax.set_title(lab, pad=3); ax.set_xticks(xs); ax.set_xlim(-0.25, 3.25)
    ax.set_xlabel('σ (×10$^{-4}$)')
axA[0].set_ylabel('Test accuracy (%)')
axA[0].set_ylim(75, 95); axA[1].set_ylim(20, 85); axA[2].set_ylim(80, 95)

# ---------------- (b) switching survives only for fast charging ----------------
for c0, col, mk in C0S:
    axB.plot(xs, [run(c0, s)['switching_test']['switches_mean'] for _, s in SIG], '-' + mk, color=col,
             ms=3.5, lw=1.0, mec='k', mew=0.4)
axB.set_xticks(xs); axB.set_xlim(-0.25, 3.25); axB.set_ylim(0, 7)
axB.set_xlabel('σ (×10$^{-4}$)'); axB.set_ylabel('Switches per neuron (mean)')

# ---------------- (c) noise-free first-switch time vs charging time ----------------
tau = [run(c0, '0')['tau_el_pred_ns'] / 1000 for c0, _, _ in C0S]
for (c0, col, mk), ta in zip(C0S, tau):
    sw = run(c0, '0')['switching_test']
    axC.plot([ta], [sw['t_first_median_ink_ns'] / 1000], mk, color=col, ms=4.5, mec='k', mew=0.4)
    axC.plot([ta], [sw['t_first_median_bg_ns'] / 1000], mk, color=col, ms=4.5, mfc='white', mec=col, mew=1.0)
axC.set_xlim(0.5, 2.3); axC.set_ylim(0, 4.5); axC.set_xticks([0.5, 1.0, 1.5, 2.0])
axC.set_xlabel('τ$_{\\rm el}$ (µs)'); axC.set_ylabel('First switch, σ = 0 (µs)')

# ---------------- (d) dt dependence of switching suppression (C0 × 1.0 and × 0.5) ----------------
for c0, col, mk in C0S[:2]:
    for dt, ls in ((10, '-'), (5, '--')):
        rows = sorted([r for r in DG if r['c0'] == c0 and r['dt'] == dt], key=lambda r: r['sigma_eq'])
        axD.plot([1e4 * r['sigma_eq'] for r in rows], [100 * r['frac_ge3_switches'] for r in rows],
                 ls, marker=mk, color=col, ms=3.2, lw=1.0, mec='k', mew=0.4)
axD.set_xlim(-0.15, 3.15); axD.set_ylim(-3, 103); axD.set_xticks([0, 1, 2, 3])
axD.set_xlabel('σ$_{\\rm eq}$ (×10$^{-4}$)'); axD.set_ylabel('Neurons with ≥ 3 switches (%)')

for ax in axA + [axB, axC, axD]:
    ax.yaxis.grid(True, color='0.9', lw=0.4); ax.set_axisbelow(True)

# ---------------- legends below ----------------
hC0 = [Line2D([], [], color=col, marker=mk, mec='k', mew=0.4, ms=4, lw=1.0,
              label=f'C$_0$ × {c0} (τ$_{{\\rm el}}$ = {1.41632 * c0:.2f} µs)') for c0, col, mk in C0S]
fig.canvas.draw()
_y = axA[0].get_position().y0 - 0.085
fig.legend(handles=hC0, loc='upper center', bbox_to_anchor=(0.5, _y), ncol=3, frameon=False,
           handlelength=2.0, columnspacing=1.6)
axC.legend(handles=[Line2D([], [], marker='o', ls='', color='k', mec='k', ms=4.5, label='ink pixels'),
                    Line2D([], [], marker='o', ls='', mfc='white', mec='k', ms=4.5, mew=1.0, label='background')],
           loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=1, frameon=False)
axD.legend(handles=[Line2D([], [], color='k', ls='-', lw=1.0, label='dt = 10 ns'),
                    Line2D([], [], color='k', ls='--', lw=1.0, label='dt = 5 ns')],
           loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False)

IOP.save(fig, f'{OUT}/fig5_noise', IOP.DOUBLE_MM,
         labels=[(axA[0], 'a', -11), (axB, 'b', -11), (axC, 'c', -11), (axD, 'd', -13)])

for c0, _, _ in C0S:
    print(f'C0×{c0}: ' + ' | '.join(
        f'σ={s}: ' + '/'.join(f'{100*run(c0, s)["test_acc"][t]:.2f}' for t, _ in TASKS) +
        f' sw={run(c0, s)["switching_test"]["switches_mean"]:.2f}' for _, s in SIG))
