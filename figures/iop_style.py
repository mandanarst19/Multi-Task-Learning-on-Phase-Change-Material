"""Shared figure style for Paper 1 — IOP Publishing (Neuromorphic Computing and Engineering).

IOP rules: single column 8.5 cm, double column 15 cm; text 8–12 pt at final size; vector PDF;
Helvetica-type font; panel labels "(a)"; information must survive greyscale (colour + marker/hatch).
"""
import io, matplotlib as mpl

MM = 1 / 25.4
SINGLE_MM, DOUBLE_MM = 85.0, 150.0
TASK_COL = {'digit': '#0072B2', 'color': '#D55E00', 'parity': '#009E73'}   # Okabe–Ito
TASK_MK = {'digit': 'o', 'color': 's', 'parity': '^'}                        # greyscale-safe markers
TASK_HATCH = {'digit': '', 'color': '////', 'parity': '....'}                # greyscale-safe bar fills
# non-task quantities: distinct hues that do not repeat the task colours (dark -> light in greyscale)
C0_COL = {0.5: '#5E3C99', 1.0: '#E7298A', 1.5: '#E6AB02'}                   # purple, magenta, gold
AUX = '#5E3C99'; AUX_LIGHT = '#CBB8E0'; SEQ_CMAP = 'Purples'                 # secondary data / maps


def apply():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['TeX Gyre Heros', 'Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans'],
        'mathtext.fontset': 'custom', 'mathtext.rm': 'TeX Gyre Heros', 'mathtext.it': 'TeX Gyre Heros:italic',
        'mathtext.bf': 'TeX Gyre Heros:bold',
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'hatch.linewidth': 0.5,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 600})


def label(fig, ax, letter, dx_mm=-9.0, dy_mm=1.0):
    """Panel label '(a)' placed a fixed distance (mm) left of / above the axes' top-left corner."""
    fig.canvas.draw()
    bb = ax.get_position(); W, H = fig.get_size_inches() / MM
    fig.text(bb.x0 + dx_mm / W, bb.y1 + dy_mm / H, f'({letter})', fontsize=9, va='bottom', ha='left')


def _tight_width_mm(fig, pad):
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    return (bb.width + 2 * pad) / MM


def save(fig, base, width_mm, pad=0.02, labels=None):
    """Resize the figure until its cropped (tight) width equals width_mm (±0.3 mm), then save
    pdf/svg/png. Fonts stay at their pt size. `labels` = [(ax, 'a', dx_mm), ...] is (re)placed after
    the final resize so panel labels keep a fixed physical offset."""
    def place():
        for t in list(getattr(fig, '_iop_labels', [])): t.remove()
        fig._iop_labels = []
        for ax, l, dx in (labels or []):
            n0 = len(fig.texts); label(fig, ax, l, dx_mm=dx); fig._iop_labels.append(fig.texts[n0])
    for _ in range(8):
        place()
        w = _tight_width_mm(fig, pad)
        if abs(w - width_mm) < 0.3: break
        s = fig.get_size_inches(); fig.set_size_inches(s * (width_mm / w))
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(f'{base}.{ext}', bbox_inches='tight', pad_inches=pad)
    h = (fig.get_tightbbox(fig.canvas.get_renderer()).height + 2 * pad) / MM
    print(f'  saved {base}.pdf  final size {w:.1f} × {h:.1f} mm (target width {width_mm:.0f} mm)')
