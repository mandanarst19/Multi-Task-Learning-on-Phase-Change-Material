"""Fig. A1 — R–T hysteresis of the VO2 device model ([Zhang], model.py), major and minor loop.

Input : rt_loop.json  (from rt_loop_kaggle.py)
Output: figA1_rt_loop.pdf / .svg / .png
Usage : python make_figA1.py rt_loop.json OUTDIR
"""
import sys, json, numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import iop_style as IOP
IOP.apply()
MM = IOP.MM
JS, OUT = sys.argv[1], sys.argv[2]
D = json.load(open(JS))
COL_MAJ, COL_MIN = IOP.AUX, IOP.C0_COL[1.0]          # purple (major), magenta (minor); no grey


def arrays(k):
    return np.array(D[k]['T_K']), np.array(D[k]['R_kOhm'])


def arrow(ax, T, R, i, col):
    ax.annotate('', xy=(T[i + 8], R[i + 8]), xytext=(T[i], R[i]),
                arrowprops=dict(arrowstyle='-|>', color=col, lw=0.9, mutation_scale=7))


fig, ax = plt.subplots(figsize=(85 * MM, 62 * MM))
Tm, Rm = arrays('major'); h = len(Tm) // 2
ax.plot(Tm[:h], Rm[:h], '-', color=COL_MAJ, lw=1.3, zorder=3)
ax.plot(Tm[h:], Rm[h:], '--', color=COL_MAJ, lw=1.3, zorder=3)
for i in (int(0.48 * h), h + int(0.47 * h)):
    arrow(ax, Tm, Rm, i, COL_MAJ)

Tn, Rn = arrays('minor_338')
# minor loop drawn under the major loop: its reheating branch lies on the major heating branch
n = len(Tn) // 3
ax.plot(Tn, Rn, '-', color=COL_MIN, lw=2.6, alpha=0.55, zorder=2, solid_capstyle='butt')
arrow(ax, Tn, Rn, n + n // 3, COL_MIN)

Tb = D['params']['T_base']
ax.axvline(Tb, color='k', ls=':', lw=0.7)
ax.set_yscale('log'); ax.set_xlim(305, 370); ax.set_ylim(1, 200)
ax.set_yticks([1, 10, 100]); ax.set_yticklabels(['1', '10', '100'])
ax.set_xlabel('Temperature (K)'); ax.set_ylabel('VO$_2$ resistance (kΩ)')
ax.yaxis.grid(True, color='0.9', lw=0.4, which='major'); ax.set_axisbelow(True)

ax.legend(handles=[Line2D([], [], color=COL_MAJ, lw=1.3, ls='-', label='major loop, heating'),
                   Line2D([], [], color=COL_MAJ, lw=1.3, ls='--', label='major loop, cooling'),
                   Line2D([], [], color=COL_MIN, lw=2.6, alpha=0.55, label='minor loop (reversal at 338 K)'),
                   Line2D([], [], color='k', ls=':', lw=0.7, label=f'substrate $T_{{\\rm base}}$ = {Tb} K')],
          loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False,
          handlelength=2.2, columnspacing=1.2)
IOP.save(fig, f'{OUT}/figA1_rt_loop', IOP.SINGLE_MM)

c = D['checks']; p = D['params']
print(f"R(325 K, heating) = {c['R_325_heating_kOhm']:.2f} kΩ | mid heating {c['T_mid_heating_K']:.1f} K, "
      f"cooling {c['T_mid_cooling_K']:.1f} K, width {c['T_mid_heating_K'] - c['T_mid_cooling_K']:.2f} K | "
      f"Tc {p['Tc']:.2f}, w {p['width_w']:.2f}, Rm {p['Rm_kOhm']:.3f} kΩ")
