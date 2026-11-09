"""Fig. 3 — temporal emergence of task information and the onset of switching (sigma = 0).

Inputs : paper1_data.json  (sigma0_60k_final, baselines_60k)
         fig2_sim.npz      (sigma = 0 switching simulation, 1000 test images)
Output : fig3_temporal.pdf / .svg / .png
         snapshots_img78.npz (coupling0_kaggle.py: image 78, coupled vs uncoupled, σ = 0)
Usage  : python make_fig3.py paper1_data.json fig2_sim.npz snapshots_img78.npz OUTDIR
"""
import sys, json, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

JS, NPZ, SNP, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
MM = 1 / 25.4
TASK_COL = {'digit': '#0072B2', 'color': '#D55E00', 'parity': '#009E73'}
TASKS = [('digit', 'Digit'), ('color', 'Colour'), ('parity', 'Parity')]
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import iop_style as IOP
IOP.apply()

J = json.load(open(JS)); F = J['sigma0_60k_final']; A = J['baselines_60k']['A_raw_input_linear']
z = np.load(NPZ, allow_pickle=False); meta = json.loads(str(z['meta']))
VMIN, VMAX = meta['V']; DT, TMAX = meta['dt'], meta['t_max']
EDGES = np.arange(0, TMAX / 1000 + 1e-9, 0.5)
T_FIRST = J['switching_sigma0']['earliest_switch_ns'] / 1000

fig = plt.figure(figsize=(150 * MM, 150 * MM))
outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.24], hspace=0.78)
gs = outer[0].subgridspec(1, 4, width_ratios=[0.85, 1.3, 0.22, 1.0], wspace=0.62)
axA, axB, axC = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[3])

# ---------------- (a) cumulative windows [0, t_end) ----------------
CUM = [('cum 0-1us', 1), ('bins 0-3', 2), ('cum 0-3us', 3), ('early 0-7', 4), ('cum 0-6us', 6), ('full (0-19)', 10)]
for k, wn in enumerate(CUM): assert F['windows'][wn[0]]['time_us'] == [0.0, float(wn[1])], wn
tend = [t for _, t in CUM]
for t, _ in TASKS:
    axA.plot(tend, [100 * F['windows'][w]['test_acc'][t] for w, _ in CUM], '-', marker=IOP.TASK_MK[t],
             ms=3, lw=1.0, color=TASK_COL[t])
    axA.axhline(100 * A[t], color=TASK_COL[t], lw=0.7, ls=(0, (3, 2)))
    axA.plot([10.3], [100 * A[t]], marker=IOP.TASK_MK[t], ms=3, mfc='white', mec=TASK_COL[t], mew=0.8,
             clip_on=False, zorder=5)
axA.axvline(T_FIRST, color='0.5', lw=0.6, ls=':')
axA.set_xlim(0, 10.3); axA.set_xticks([0, 2, 4, 6, 8, 10])
axA.set_ylim(0, 100); axA.set_xlabel('End of window (µs)'); axA.set_ylabel('Test accuracy (%)')

# ---------------- (b) sliding 1-µs windows with switching onsets behind ----------------
sl = F['sliding_1us']; c = np.array(sl['centre_us'])
on = z['onsetB'].astype(float); per_bin = on.reshape(len(EDGES) - 1, -1).sum(1) / on.sum()
tf = z['tfirstB'].astype(float).ravel() / 1000; ok = tf >= 0
first = np.histogram(tf[ok], bins=EDGES)[0] / ok.sum()
axB2 = axB.twinx()
axB2.bar(EDGES[:-1], 100 * per_bin, width=0.5, align='edge', color=IOP.AUX_LIGHT, lw=0, zorder=0)
axB2.step(EDGES[:-1], 100 * first, where='post', color=IOP.AUX, lw=0.8, zorder=1)
axB2.set_ylim(0, 100); axB2.set_ylabel('Switching (%)', color=IOP.AUX, labelpad=2)
axB2.tick_params(axis='y', colors=IOP.AUX)
axB.set_zorder(axB2.get_zorder() + 1); axB.patch.set_visible(False)
for t, _ in TASKS:
    axB.plot(c, [100 * a[t] for a in sl['acc']], '-', marker=IOP.TASK_MK[t], ms=2.4, lw=0.9, color=TASK_COL[t])
axB.axvline(T_FIRST, color='0.5', lw=0.6, ls=':')
axB.set_xlim(0, 10); axB.set_xticks(range(0, 11, 2)); axB.set_ylim(0, 100)
axB.set_xlabel('Window centre (µs)'); axB.set_ylabel('Test accuracy (%)')

# ---------------- (c) first switch vs. input voltage ----------------
V = z['VB'].astype(float).ravel()
ink = ok & (V > VMIN + 1e-6); bg = ok & ~(V > VMIN + 1e-6)
hb = axC.hexbin(V[ink], tf[ink], gridsize=(34, 26), extent=(VMIN, VMAX, 1.9, 3.0), bins='log',
                cmap=IOP.SEQ_CMAP, mincnt=1, linewidths=0, rasterized=True)
e = np.linspace(VMIN, VMAX, 18); m = 0.5 * (e[1:] + e[:-1])
med = [np.median(tf[ink & (V >= lo) & (V < hi)]) if (ink & (V >= lo) & (V < hi)).sum() > 50 else np.nan
       for lo, hi in zip(e[:-1], e[1:])]
axC.plot(m, med, color='k', lw=1.0)
q = np.percentile(tf[bg], [25, 50, 75])
axC.errorbar([VMIN], [q[1]], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt='o', ms=3, color='k',
             capsize=1.5, lw=0.7, clip_on=False, zorder=5)
axC.set_xlim(VMIN - 0.03, VMAX); axC.set_ylim(1.9, 3.0); axC.set_xticks([10.5, 11.0, 11.5, 12.0])
axC.set_xlabel('Input voltage $V$ (V)'); axC.set_ylabel('First switch (µs)')
cb = fig.colorbar(hb, ax=axC, pad=0.03, fraction=0.06); cb.set_label('Pixels', fontsize=8, labelpad=1)
cb.ax.tick_params(labelsize=8)


# ---------------- (d) current across the array at four instants, one test image ----------------
from matplotlib.colors import PowerNorm
b9 = int(np.argmax(z['VA'].max(1)))                                  # test image 78 (digit 9, Yellow)
S = np.load(SNP, allow_pickle=False); SNAP_NS = [int(t) for t in S['t_ns']]
assert SNAP_NS == [1900, 2100, 2500, 2850] and np.array_equal(S['V'], z['VA'][b9])
# coupled row from the snapshot file agrees with fig2_sim.npz (float16 storage) -> both rows from one source
assert all(np.abs(S['coupled_maps'][k] - z['IA'][t // DT, b9].astype(float)).max() < 5e-3 for k, t in enumerate(SNAP_NS))
ROWS = [('coupled_maps', 'coupled\n(c = 0.02)'), ('uncoupled_maps', 'uncoupled\n(c = 0)')]
gsd = outer[1].subgridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.07], wspace=0.12, hspace=0.12)
axS = [[fig.add_subplot(gsd[r, k]) for k in range(4)] for r in range(2)]
nrm = PowerNorm(gamma=0.35, vmin=0, vmax=5.5)
for r, (key, lab) in enumerate(ROWS):
    for k, (ax_, tns) in enumerate(zip(axS[r], SNAP_NS)):
        ims = ax_.imshow(S[key][k].reshape(28, 28), cmap='inferno', norm=nrm, interpolation='nearest')
        ax_.set_xticks([]); ax_.set_yticks([])
        if r == 0: ax_.set_title(f'{tns / 1000:.2f} µs', pad=3)
        if k == 0: ax_.set_ylabel(lab, fontsize=8, labelpad=3)
cbs = fig.colorbar(ims, cax=fig.add_subplot(gsd[:, 4]), ticks=[0, 0.2, 1, 3, 5])
cbs.set_label('$I$ (mA)'); cbs.ax.tick_params(labelsize=8)

# ---------------- shared legend below ----------------
h = [Line2D([], [], color=TASK_COL[t], marker=IOP.TASK_MK[t], ms=3, lw=1.0, label=l) for t, l in TASKS]
h += [Line2D([], [], color='0.3', lw=0.7, ls=(0, (3, 2)), label='raw input, linear (a; open marker)'),
      Line2D([], [], color='0.5', lw=0.6, ls=':', label='first switch, 2.03 µs'),
      Patch(color=IOP.AUX_LIGHT, label='all switch events (b)'),
      Line2D([], [], color=IOP.AUX, lw=0.8, label='first switch per neuron (b)'),
      Line2D([], [], color='k', lw=1.0, label='median, ink pixels (c)'),
      Line2D([], [], color='k', marker='o', ms=3, lw=0, label='background, median ± IQR (c)')]
fig.canvas.draw(); _y = axA.get_position().y0 - 0.075
fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, _y), ncol=3, frameon=False,
           handlelength=1.6, columnspacing=1.2)

IOP.save(fig, f'{OUT}/fig3_temporal', IOP.DOUBLE_MM, labels=[(axA, 'a', -10), (axB, 'b', -10), (axC, 'c', -11), (axS[0][0], 'd', -11)])

print('cumulative (end µs: digit/colour/parity):')
for w, t in CUM:
    a = F['windows'][w]['test_acc']; print(f'  {t:>2} µs  {100*a["digit"]:.2f} / {100*a["color"]:.2f} / {100*a["parity"]:.2f}')
print(f'raw input (A): {100*A["digit"]:.2f} / {100*A["color"]:.2f} / {100*A["parity"]:.2f}')
