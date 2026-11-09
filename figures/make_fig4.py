"""Fig. 4 — colour errors are set by input-amplitude resolution (sigma = 0, 60k, P5).

Input : paper1_data.json  (sigma0_60k_final, dataset, logistic_sigma0_60k)
Output: fig4_colour.pdf / .svg / .png
Usage : python make_fig4.py paper1_data.json OUTDIR
"""
import sys, json, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

JS, OUT = sys.argv[1], sys.argv[2]
MM = 1 / 25.4
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import iop_style as IOP
IOP.apply()
COL = '#D55E00'

J = json.load(open(JS)); F = J['sigma0_60k_final']; DS = J['dataset']
NAMES = F['color_order']; assert NAMES == DS['colour_names']
P = np.array(DS['palette_rgb'], float); L = P @ np.array([0.299, 0.587, 0.114])
V = 10.5 + 1.7 * L                                            # full-intensity pixel voltage
CM = np.array(F['confusion_color'], float)
o = np.argsort(L); N = [NAMES[i] for i in o]
CMo = CM[np.ix_(o, o)]; R = 100 * CMo / CMo.sum(1, keepdims=True)   # row-normalised, %
acc = np.array([F['per_class_color'][n] for n in NAMES]) * 100
LG = J['logistic_sigma0_60k']['reservoir']['color']
acc_lg = np.array([LG['per_class_color'][n] for n in NAMES]) * 100
dV = np.array([1e3 * np.min(np.abs(np.delete(V, i) - V[i])) for i in range(10)])  # mV to nearest colour
pairs = F['pairs']; in_pair = {n for p in pairs for n in p}

fig = plt.figure(figsize=(150 * MM, 74 * MM))
gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 0.95], wspace=0.75)
ax = fig.add_subplot(gs[0]); axb = fig.add_subplot(gs[1])

# ---------------- (a) confusion matrix, classes ordered by luminance ----------------
im = ax.imshow(R, cmap=IOP.SEQ_CMAP, vmin=0, vmax=100, interpolation='nearest')
pos = {n: k for k, n in enumerate(N)}
for a, b in pairs:
    i, j = sorted((pos[a], pos[b]))                               # adjacent in luminance order
    ax.add_patch(Rectangle((i - 0.5, i - 0.5), 2, 2, fill=False, ec=COL, lw=1.2))
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(N, rotation=60, ha='right', rotation_mode='anchor'); ax.set_yticklabels(N)
for tl in ax.get_xticklabels() + ax.get_yticklabels():
    if tl.get_text() in in_pair: tl.set_color(COL)
ax.set_xlabel('Predicted colour'); ax.set_ylabel('True colour')
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label('Share of true class (%)', fontsize=8)
cb.ax.tick_params(labelsize=8)

# ---------------- (b) per-class accuracy vs. voltage gap to the nearest colour ----------------
# display-only jitter for colours with identical ΔV (the four pair colours; Magenta/Gray)
jit = np.ones(10)
for v in np.unique(np.round(dV, 2)):
    grp = [i for i in range(10) if round(dV[i], 2) == v]
    for k, i in enumerate(grp): jit[i] = 1.0 + 0.06 * (k - (len(grp) - 1) / 2)
for i, n in enumerate(NAMES):
    x = dV[i] * jit[i]; ec = COL if n in in_pair else 'k'; lw = 1.1 if n in in_pair else 0.5
    axb.plot([x, x], [acc[i], acc_lg[i]], color='#9E9AC8', lw=0.6, zorder=2)
    axb.scatter(x, acc[i], s=26, marker='o', color=P[i], edgecolors=ec, linewidths=lw, zorder=3)
    axb.scatter(x, acc_lg[i], s=26, marker='D', color=P[i], edgecolors=ec, linewidths=lw, zorder=3)
axb.set_xscale('log'); axb.set_xlim(5, 400); axb.set_ylim(40, 103)
axb.set_xticks([10, 30, 100, 300]); axb.set_xticklabels(['10', '30', '100', '300'])
axb.set_xlabel('Δ$V$ to nearest colour, full-intensity pixel (mV)')
axb.set_ylabel('Per-class colour accuracy (%)')
axb.yaxis.grid(True, color='0.9', lw=0.4); axb.set_axisbelow(True)
h = [Line2D([], [], marker='o', ls='', ms=5, mfc='white', mec='k', mew=0.5, label='ridge readout (linear, as in a)'),
     Line2D([], [], marker='D', ls='', ms=4.5, mfc='white', mec='k', mew=0.5, label='logistic readout'),
     Line2D([], [], marker='o', ls='', ms=5, mfc='white', mec=COL, mew=1.2,
            label='near-equal-luminance pair (orange boxes in a)')]
axb.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=1, frameon=False)

IOP.save(fig, f'{OUT}/fig4_colour', IOP.DOUBLE_MM, labels=[(ax, 'a', -18), (axb, 'b', -11)])

print('colour  V(V)    dV_nearest(mV)  ridge(%)  logistic(%)')
for i in o: print(f'{NAMES[i]:8s} {V[i]:.4f}  {dV[i]:7.2f}        {acc[i]:.1f}     {acc_lg[i]:.1f}')
off = CM.sum() - np.trace(CM)
print(f'errors in the two pairs: {F["colour_err_in_pairs"]*100:.1f}% of all colour errors ({int(off)} errors)')
