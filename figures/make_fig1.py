"""Fig. 1 — task, encoding, device and readout.

Input : fig2_sim.npz (Kaggle simulation; holds the input voltages and traces of the 10 Fig-1a images)
Output: fig1_overview.pdf / .svg / .png
Usage : python make_fig1.py fig2_sim.npz OUTDIR
Panels (a) and (b) are rebuilt from the stored input voltages: pixel = (V - Vmin) / ((Vmax - Vmin) L_c),
which reproduces the Kaggle panels exactly without needing MNIST here.
"""
import sys, json, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import schemdraw, schemdraw.elements as elm

SRC, OUT = sys.argv[1], sys.argv[2]
MM = 1 / 25.4
TASK_COL = {'digit': '#0072B2', 'color': '#D55E00', 'parity': '#009E73'}
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import iop_style as IOP
IOP.apply()

z = np.load(SRC, allow_pickle=False)
meta = json.loads(str(z['meta']))
P = np.array(meta['palette'], float)
VMIN, VMAX = meta['V']; DT, TMAX = meta['dt'], meta['t_max']
NAMES = ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'Orange', 'Purple', 'DkGreen', 'Gray']
L = P @ np.array([0.299, 0.587, 0.114]); Vpk = VMIN + (VMAX - VMIN) * L
ix = {n: i for i, n in enumerate(NAMES)}
PAIRS = [('Red', 'DkGreen'), ('Green', 'Orange')]
VA, YD, YC = z['VA'].astype(float), z['ydA'], z['ycA']
G = np.clip((VA - VMIN) / ((VMAX - VMIN) * L[YC][:, None]), 0, 1).reshape(-1, 28, 28)
cmapV = plt.get_cmap('magma'); normV = Normalize(VMIN, VMAX)
colV = lambda v: cmapV(0.08 + 0.72 * normV(v))

fig = plt.figure(figsize=(150 * MM, 178 * MM))
outer = fig.add_gridspec(3, 1, height_ratios=[0.5, 0.78, 1], hspace=0.30)
gsa = outer[1].subgridspec(2, 11, width_ratios=[1] * 10 + [0.12], wspace=0.08, hspace=0.30)
gsb = outer[2].subgridspec(1, 3, width_ratios=[0.86, 1.0, 1.15], wspace=0.62)

# ---------------- (a) pipeline overview ----------------
axP = fig.add_subplot(outer[0]); axP.set_xlim(0, 140); axP.set_ylim(0, 32); axP.axis('off')
kP = int(np.argmax(VA.max(1)))                                     # test image 78 (digit 9, Yellow)
BOX = dict(boxstyle='round,pad=0.3,rounding_size=0.8', fc='white', ec='0.25', lw=0.6)
def stage(x, w, title, body=None, panel=None, thumb=None, cmap=None, vmin=None, vmax=None):
    axP.add_patch(FancyBboxPatch((x, 2), w, 29, boxstyle='round,pad=0,rounding_size=1.2',
                                 fc='white', ec='0.25', lw=0.6))
    axP.text(x + w / 2, 29.6, title, ha='center', va='top', fontsize=8, fontweight='bold')
    if thumb is not None:
        ia = axP.inset_axes([x + w / 2 - 6, 9.5, 12, 14], transform=axP.transData)
        ia.imshow(thumb, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest'); ia.set_xticks([]); ia.set_yticks([])
    if body: axP.text(x + w / 2, 14.5 if thumb is None else 8.2, body, ha='center', va='center' if thumb is None else 'top',
                      fontsize=8, linespacing=1.15)
    if panel: axP.text(x + w / 2, -0.6, panel, ha='center', va='top', fontsize=8, color='k')
def arrow(x0, x1, y=16):
    axP.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=7, lw=0.8, color='0.25'))
X = [0, 23.5, 47, 70.5, 94]; Wd = 20
stage(X[0], Wd, 'Input', 'RGB image', thumb=G[kP][..., None] * P[YC[kP]][None, None, :])
stage(X[1], Wd, 'Encoding', '$V \\propto L$', '(b), (c)', thumb=VA[kP].reshape(28, 28), cmap='magma', vmin=VMIN, vmax=VMAX)
stage(X[2], Wd, 'VO$_2$ array', '28 × 28\nneuristors,\nthermally\ncoupled', '(d)')
stage(X[3], Wd, 'Response', '$I(t)$, 10 µs\n20 bins\nmax per bin', '(e)')
stage(X[4], Wd, 'Features', '784 × 20\n= 15,680\nshared', None)
for k in range(4): arrow(X[k] + Wd + 0.4, X[k + 1] - 0.4)
for yy, t, n in ((25, 'digit', 10), (16, 'color', 10), (7, 'parity', 2)):
    axP.add_patch(FancyArrowPatch((X[4] + Wd + 0.4, 16), (120, yy), arrowstyle='-|>', mutation_scale=7,
                                  lw=0.8, color=TASK_COL[t], shrinkA=0, shrinkB=0))
    axP.text(121, yy, f"{'colour' if t == 'color' else t} ({n})", ha='left', va='center', fontsize=8,
             color=TASK_COL[t], bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=TASK_COL[t], lw=0.7))
axP.text(129, -0.6, 'linear readouts\n(only trained part)', ha='center', va='top', fontsize=8, color='k', linespacing=1.1)

# ---------------- (a) RGB input and the voltage the array receives ----------------
axa = [[fig.add_subplot(gsa[r, k]) for k in range(10)] for r in range(2)]
for k in range(10):
    c = YC[k]
    axa[0][k].imshow(G[k][..., None] * P[c][None, None, :], interpolation='nearest')
    axa[0][k].set_title(f'{YD[k]} ({"even" if YD[k] % 2 == 0 else "odd"})\n{NAMES[c]}',
                        fontsize=8, pad=2, linespacing=1.1)
    im = axa[1][k].imshow(VA[k].reshape(28, 28), cmap='magma', vmin=VMIN, vmax=VMAX,
                          interpolation='nearest')
    for r in range(2):
        axa[r][k].set_xticks([]); axa[r][k].set_yticks([])
axa[0][0].set_ylabel('RGB', fontsize=8); axa[1][0].set_ylabel('Input $V$', fontsize=8)
cax = fig.add_subplot(gsa[1, 10])
cb = fig.colorbar(im, cax=cax, ticks=[10.5, 11.0, 11.5, 12.0])
cb.set_label('$V$ (V)', fontsize=8); cb.ax.tick_params(labelsize=8)

# ---------------- (b) palette on the voltage axis ----------------
axb = fig.add_subplot(gsb[0])
o = np.argsort(L); pos = {i: r for r, i in enumerate(o)}
axb.hlines(np.arange(10), VMIN, Vpk[o], color='0.85', lw=0.6, zorder=1)
axb.scatter(Vpk[o], np.arange(10), s=20, c=P[o], edgecolors='k', linewidths=0.5, zorder=3)
axb.set_yticks(np.arange(10)); axb.set_yticklabels([NAMES[i] for i in o])
axb.set_xlim(VMIN, VMAX); axb.set_ylim(-0.6, 9.6); axb.set_xticks([10.5, 11.0, 11.5, 12.0])
axb.set_xlabel('Input $V$, full-intensity pixel (V)')
sec = axb.secondary_xaxis('top', functions=(lambda v: (v - VMIN) / (VMAX - VMIN),
                                            lambda l: VMIN + (VMAX - VMIN) * l))
sec.set_xlabel('Luminance $L$'); sec.set_xticks([0, 0.5, 1])
for a, b in PAIRS:
    ra, rb = sorted((pos[ix[a]], pos[ix[b]]))
    xb = max(Vpk[ix[a]], Vpk[ix[b]]) + 0.05
    axb.plot([xb, xb + 0.03, xb + 0.03, xb], [ra, ra, rb, rb], color=TASK_COL['color'], lw=0.7)
dv = 1e3 * abs(Vpk[ix['Red']] - Vpk[ix['DkGreen']])
axb.text(0.5, -0.25, f'brackets: ΔV = {dv:.2f} mV', transform=axb.transAxes,
         ha='center', va='top', fontsize=8, color=TASK_COL['color'])

# ---------------- (c) one neuristor and the coupled array ----------------
gsc = gsb[1].subgridspec(2, 1, height_ratios=[1, 1.05], hspace=0.05)
axc1 = fig.add_subplot(gsc[0]); axc2 = fig.add_subplot(gsc[1])
d = schemdraw.Drawing(canvas=axc1)
d.config(fontsize=8, lw=0.8, unit=2.2)
src = d.add(elm.SourceV().up().label('$V_{ij}$', loc='bottom', ofst=0.15))
d.add(elm.Resistor().right().label('$R_0$', ofst=0.1))
d.add(elm.Dot())
d.push()
d.add(elm.Capacitor().down().label('$C_0$', loc='bottom', ofst=0.1))
d.add(elm.Line().left().tox(src.start))
d.pop()
d.add(elm.Line().right(1.6))
d.add(elm.ResistorVar().down().label('VO$_2$', loc='bottom', ofst=0.1))
d.add(elm.Line().left(1.6))
d.add(elm.Ground().at(src.start))
d.draw(show=False)
axc1.set_aspect('equal'); axc1.axis('off')

n = 3; s = 1.0; g = 0.85
hi = (1, 1)
for i in range(n):
    for j in range(n):
        fc = colV(11.6) if (i, j) == hi else '#cfd8e3'
        axc2.add_patch(FancyBboxPatch((j * (s + g), -i * (s + g)), s, s,
                                      boxstyle='round,pad=0,rounding_size=0.15',
                                      fc=fc, ec='0.35', lw=0.5))
cx, cy = hi[1] * (s + g) + s / 2, -hi[0] * (s + g) + s / 2
for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    x0, y0 = cx + dx * (s / 2 + 0.02), cy + dy * (s / 2 + 0.02)
    x1, y1 = cx + dx * (s / 2 + g - 0.02), cy + dy * (s / 2 + g - 0.02)
    axc2.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='<|-|>', mutation_scale=7,
                                   lw=0.9, color='#B2182B', shrinkA=0, shrinkB=0))
W = n * s + (n - 1) * g
axc2.add_patch(Rectangle((-0.35, -(n - 1) * (s + g) - 0.35), W + 0.7, W + 0.7, fc='none',
                         ec='0.55', lw=0.5, ls=(0, (2, 1.5))))
axc2.text(W / 2, -(n - 1) * (s + g) - 0.6, 'Al$_2$O$_3$ substrate · 28 × 28 array',
          ha='center', va='top', fontsize=8)
axc2.text(W / 2, s + 0.55, 'heat flows to nearest neighbours', ha='center', va='bottom',
          fontsize=8, color='#B2182B')
axc2.set_xlim(-0.6, W + 0.6); axc2.set_ylim(-(n - 1) * (s + g) - 1.6, s + 1.3)
axc2.set_aspect('equal'); axc2.axis('off')

# ---------------- (d) feature extraction from one neuron ----------------
gsd = gsb[2].subgridspec(3, 1, height_ratios=[0.10, 1, 0.42])
axd1 = fig.add_subplot(gsd[1])
k = int(np.argmax(VA.max(1))); nbr = int(np.argmax(VA[k]))
I = z['IA'][:, k, nbr].astype(float); t = np.arange(len(I)) * DT / 1000
edges = np.arange(0, TMAX / 1000 + 1e-9, 0.5)
for e0 in edges[:-1:2]:
    axd1.axvspan(e0, e0 + 0.5, color='0.93', lw=0, zorder=0)
axd1.plot(t, I, color=colV(VA[k, nbr]), lw=0.8, zorder=2)
Ib = I.reshape(len(edges) - 1, -1); am = Ib.argmax(1)
tm = (np.arange(len(edges) - 1) * Ib.shape[1] + am) * DT / 1000
axd1.plot(tm, Ib.max(1), 'o', ms=2.2, color='k', zorder=3)
axd1.set_xlim(0, TMAX / 1000); axd1.set_ylim(0, 6); axd1.set_xticks([0, 2, 4, 6, 8, 10])
axd1.set_xlabel('Time (µs)'); axd1.set_ylabel('$I$ (mA)')
axd1.annotate('shading: 20 bins × 0.5 µs\ndots: max per bin = feature', xy=(0.5, 0), xycoords='axes fraction',
              xytext=(0, -27), textcoords='offset points', ha='center', va='top', fontsize=8)



IOP.save(fig, f'{OUT}/fig1_overview', IOP.DOUBLE_MM,
         labels=[(axP, 'a', -4), (axa[0][0], 'b', -12), (axb, 'c', -17), (axc1, 'd', -3), (axd1, 'e', -10)])
print(f'(d) trace: image {int(z["idxA"][k])}, neuron {nbr}, V = {VA[k, nbr]:.3f} V')
print(f'(b) ΔV = {dv:.2f} mV')
