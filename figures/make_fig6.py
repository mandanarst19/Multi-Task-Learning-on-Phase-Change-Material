"""Fig. 6 — thermal-coupling control and sweep. Only the neighbour heat flow S_couple = c·Sth is varied;
the heat loss to the substrate is fixed at 0.92 Sth. Same 20k P5 protocol, C0 × 1.0, 10,000 test images.

Input : paper1_data.json  (c0_sigma_grid_20k, coupling_control_20k, coupling_sweep_20k, baseline_D_logistic_and_20k, dataset, sigma0_60k_final)
Output: fig6_coupling.pdf / .svg / .png
Usage : python make_fig6.py paper1_data.json OUTDIR
"""
import sys, json, numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import iop_style as IOP
IOP.apply(); MM = IOP.MM
JS, OUT = sys.argv[1], sys.argv[2]
J = json.load(open(JS)); G = J['c0_sigma_grid_20k']['runs']; U = J['coupling_control_20k']['runs']
SW = J['coupling_sweep_20k']['runs']
TASKS = [('digit', 'Digit'), ('color', 'Colour'), ('parity', 'Parity')]
PAIR_COL = '#D55E00'
PG = J['c0_sigma_grid_20k']['protocol']
for r in [U['uncoupled_sigma0'], U['uncoupled_sigma2e-4']] + list(SW.values()):   # one protocol everywhere
    pr = r['protocol']; assert not pr['smoke']
    for k in ('n_train', 'n_val', 'n_test', 'split_seed', 'dt', 'nb', 'batch'): assert pr[k] == PG[k], k
assert all(abs(SW['c0.020_sigma0']['test_acc'][t] - G['c0_1.0_sigma0']['test_acc'][t]) < 5e-4 for t, _ in TASKS)

# sweep at σ = 0 (c = 0.02 from the grid; reproduced exactly by the sweep)
CS = [(0.0, U['uncoupled_sigma0'])] + sorted([(r['cond']['couple'], r) for k, r in SW.items() if k != 'c0.020_sigma0']
                                            + [(0.02, G['c0_1.0_sigma0'])], key=lambda x: x[0])
cx = [c for c, _ in CS]

fig = plt.figure(figsize=(150 * MM, 66 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.55, 1.15], wspace=0.55)
axA, axB, axC = (fig.add_subplot(gs[k]) for k in range(3))

# ---------------- (a) accuracy vs coupling strength, σ = 0 ----------------
for t, lab in TASKS:
    axA.plot(cx, [100 * r['test_acc'][t] for _, r in CS], '-' + IOP.TASK_MK[t], color=IOP.TASK_COL[t],
             ms=4, lw=1.0, mec='white', mew=0.4)
DR = J['baseline_D_logistic_and_20k']['ridge_20k']['test_acc']           # static baseline D, same 20k protocol
for t, lab in TASKS:
    axA.axhline(100 * DR[t], color=IOP.TASK_COL[t], lw=0.8, ls=(0, (4, 2)), zorder=1)
axA.axvline(0.02, color=IOP.AUX, lw=0.7, ls=':')
axA.set_xlim(-0.002, 0.042); axA.set_xticks([0, 0.01, 0.02, 0.03, 0.04])
axA.set_xticklabels(['0', '0.01', '0.02', '0.03', '0.04'])
axA.set_xlabel('Coupling $c$ ($S_{\\rm couple}/S_{\\rm th}$)'); axA.set_ylabel('Test accuracy, σ = 0 (%)')
axA.set_ylim(75, 96); axA.set_yticks(range(75, 96, 5)); axA.yaxis.grid(True, color='0.9', lw=0.4); axA.set_axisbelow(True)

# ---------------- (b) default noise: coupled (filled) vs uncoupled (open) ----------------
RN = {'coupled': G['c0_1.0_sigma2e-4'], 'uncoupled': U['uncoupled_sigma2e-4']}
for k, (t, lab) in enumerate(TASKS):
    c_ = IOP.TASK_COL[t]; mk = IOP.TASK_MK[t]
    vc, vu = (100 * RN[m]['test_acc'][t] for m in ('coupled', 'uncoupled'))
    axB.plot([k, k], [vc, vu], '-', color=c_, lw=1.0, alpha=0.5)
    axB.plot([k], [vc], mk, ms=5, mfc=c_, mec=c_)
    axB.plot([k], [vu], mk, ms=5, mfc='white', mec=c_, mew=1.1)
axB.set_xlim(-0.6, 2.6); axB.set_xticks(range(3)); axB.set_xticklabels([l for _, l in TASKS], rotation=60,
                                                                        ha='right', rotation_mode='anchor')
axB.set_ylim(75, 96); axB.set_yticks(range(75, 96, 5)); axB.set_ylabel('Test accuracy, σ = 2×10$^{-4}$ (%)')
axB.yaxis.grid(True, color='0.9', lw=0.4); axB.set_axisbelow(True)

# ---------------- (c) per-colour accuracy at σ = 0, c = 0 vs c = 0.02 ----------------
DS = J['dataset']; NAMES = DS['colour_names']
L = np.array(DS['palette_rgb'], float) @ np.array([0.299, 0.587, 0.114]); o = np.argsort(L); N = [NAMES[i] for i in o]
pairs = J['sigma0_60k_final']['pairs']; in_pair = {n for p in pairs for n in p}
PC = {'coupled': G['c0_1.0_sigma0']['per_class_color'], 'uncoupled': U['uncoupled_sigma0']['per_class_color']}
x = np.arange(10)
for i, n in enumerate(N):
    axC.plot([i, i], [100 * PC['coupled'][n], 100 * PC['uncoupled'][n]], '-', color=IOP.AUX_LIGHT, lw=1.6, zorder=2)
for m, mfc in (('coupled', IOP.AUX), ('uncoupled', 'white')):
    axC.plot(x, [100 * PC[m][n] for n in N], 'o', ms=4.2, mfc=mfc, mec=IOP.AUX, mew=1.0, zorder=3)
pos = {n: k for k, n in enumerate(N)}
for p in pairs:
    i, j = sorted(pos[n] for n in p)
    axC.axvspan(i - 0.42, j + 0.42, color=PAIR_COL, alpha=0.10, lw=0, zorder=0)
axC.set_xticks(x); axC.set_xticklabels(N, rotation=60, ha='right', rotation_mode='anchor')
for tl in axC.get_xticklabels():
    if tl.get_text() in in_pair: tl.set_color(PAIR_COL)
axC.set_xlim(-0.6, 9.6); axC.set_ylim(0, 105); axC.set_ylabel('Colour accuracy, σ = 0 (%)')
axC.yaxis.grid(True, color='0.9', lw=0.4); axC.set_axisbelow(True)

# ---------------- legend below ----------------
h = [Line2D([], [], marker=IOP.TASK_MK[t], ls='-', lw=1.0, ms=4.5, color=IOP.TASK_COL[t], label=l) for t, l in TASKS]
h += [Line2D([], [], color='k', lw=0.8, ls=(0, (4, 2)), label='baseline D, same protocol (a)'),
      Line2D([], [], color=IOP.AUX, lw=0.7, ls=':', label='default c = 0.02 (a)'),
      Line2D([], [], marker='o', ls='', ms=4.5, mfc='k', mec='k', label='coupled, c = 0.02 (b, c)'),
      Line2D([], [], marker='o', ls='', ms=4.5, mfc='white', mec='k', mew=1.1, label='uncoupled, c = 0 (b, c)'),
      Patch(fc=PAIR_COL, alpha=0.10, label='pairs 9.35 mV apart (c)')]
fig.canvas.draw(); _y = axA.get_position().y0 - 0.235
fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, _y), ncol=3, frameon=False,
           handlelength=1.6, columnspacing=1.2)
IOP.save(fig, f'{OUT}/fig6_coupling', IOP.DOUBLE_MM, labels=[(axA, 'a', -11), (axB, 'b', -11), (axC, 'c', -11)])

for c, r in CS:
    a = r['test_acc']; print(f'c = {c:.3f}: {100*a["digit"]:.2f} / {100*a["color"]:.2f} / {100*a["parity"]:.2f}')
