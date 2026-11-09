# =============================================================================
# C0 × σ grid — P5 protocol, n_train = 20k — STANDALONE Kaggle cell (Paper 1, Fig. 5a)
# =============================================================================
# Needs in the notebook inputs: model.py (Zhang code) and paper1_data.json. Internet ON (MNIST).
# Accelerator: GPU T4 recommended (simulation on GPU, ridge readout on CPU float64).
#
# HOW TO RUN
#   1) SMOKE = True  → run the cell interactively (~5–10 min). It checks every step on a tiny
#      subset and prints a runtime estimate. Send me the printed output.
#   2) SMOKE = False → Save & Run All (commit). Each condition writes its own JSON to
#      /kaggle/working/c0sigma/ and is skipped if already done, so a crash loses at most one.
#
# WHY A FULL GRID (not only C0×0.5, σ=0): the original T1 notebook is lost, so its exact train
# subset / validation split cannot be reproduced. Mixing a new cell with old cells would compare
# two protocols. This cell runs every condition under ONE protocol; the two σ=3e-4 cells whose
# original result files exist (c0_1.0, c0_0.5) act as a reproduction check of the protocol.
# =============================================================================
import os, sys, gc, json, time, importlib, numpy as np, torch
from torchvision import datasets

SMOKE = True

# ---------------- conditions, in priority order (name, C0 factor, σ in code units) -------------
RUN_SET = [
    ('c0_0.5_sigma0',    0.5, 0.0),     # the missing cell
    ('c0_1.0_sigma0',    1.0, 0.0),     # was chat-only (90.19 / 78.99 / 92.56)
    ('c0_1.5_sigma0',    1.5, 0.0),     # was printed output (90.08 / 77.90 / 92.40)
    ('c0_1.0_sigma3e-4', 1.0, 3e-4),    # reproduction check: file value 79.83 / 56.88 / 85.39
    ('c0_0.5_sigma3e-4', 0.5, 3e-4),    # reproduction check: file value 89.72 / 78.06 / 92.23
    ('c0_0.5_sigma2e-4', 0.5, 2e-4),    # was chat-only (89.98 / 78.03 / 92.52)
    ('c0_1.0_sigma2e-4', 1.0, 2e-4),    # was chat-only (85.90 / 76.75 / 89.16)
    ('c0_1.5_sigma2e-4', 1.5, 2e-4),    # was chat-only (81.93 / 38.30 / 86.18)
    ('c0_1.5_sigma3e-4', 1.5, 3e-4),    # never run
]
CFG = dict(dt=10, t_max=10_000, T_base=325, R0=12.0, cth=0.15, couple=0.02, Vmin=10.5, Vmax=12.2,
           nb=20, n_train=20_000, n_val=4_000, n_test=10_000, split_seed=42, noise_seed=0, batch=500)
if SMOKE:
    RUN_SET = [RUN_SET[1], RUN_SET[0]]            # C0×1.0 first: its statistics are known
    CFG.update(n_train=300, n_val=100, n_test=200, batch=100)
OUT = '/kaggle/working/c0sigma' + ('_smoke' if SMOKE else ''); os.makedirs(OUT, exist_ok=True)

# ---------------- model.py (F7: original reversal) ----------------
def find(fname):
    for base in ('/kaggle/working', '/kaggle/input'):
        for r, _, fs in os.walk(base):
            for f in fs:
                if f == fname or (fname.endswith('.json') and f.startswith(fname[:-5]) and f.endswith('.json')):
                    return os.path.join(r, f)
    raise FileNotFoundError(fname)
sys.path.insert(0, os.path.dirname(find('model.py'))); import model; importlib.reload(model)
assert model.VO2.reversal.__name__ == 'reversal'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False); torch.set_num_threads(os.cpu_count())
print('device for simulation:', DEV, '| CPU threads for readout:', os.cpu_count())

# ---------------- dataset (verified reconstruction, see paper1_data.json → dataset) ----------------
J = json.load(open(find('paper1_data.json')))
NAMES = J['sigma0_60k_final']['color_order']          # works with the old and the new JSON
assert NAMES == ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'Orange', 'Purple', 'DkGreen', 'Gray']
P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
              [1, .5, 0], [.5, 0, 1], [0, .5, 0], [.5, .5, .5]], np.float32)   # verified palette
L = P @ np.array([0.299, 0.587, 0.114], np.float32)
_lum = J['baselines_60k']['digit_acc_by_colour_sigma2e-4_seed0']
for _n, _l in zip(_lum['order_by_luminance'], _lum['luminance']):
    assert abs(L[NAMES.index(_n)] - _l) < 6e-4, _n                       # palette matches the run
np.random.seed(42); YC_all = np.random.randint(0, 10, 60000); YC_te_all = np.random.randint(0, 10, 10000)
assert (np.bincount(YC_te_all, minlength=10) == np.array(J['sigma0_60k_final']['confusion_color']).sum(1)).all()
mn_tr = datasets.MNIST('/kaggle/working/data', train=True, download=True)
mn_te = datasets.MNIST('/kaggle/working/data', train=False, download=True)
nTR, nTE = CFG['n_train'], CFG['n_test']
G_tr = mn_tr.data[:nTR].numpy().reshape(nTR, -1).astype(np.float32) / 255     # first n_train images
G_te = mn_te.data[:nTE].numpy().reshape(nTE, -1).astype(np.float32) / 255
y_tr = {'digit': mn_tr.targets[:nTR].numpy(), 'color': YC_all[:nTR]}
y_te = {'digit': mn_te.targets[:nTE].numpy(), 'color': YC_te_all[:nTE]}
for d in (y_tr, y_te): d['parity'] = d['digit'] % 2
volts = lambda G, yc: CFG['Vmin'] + (CFG['Vmax'] - CFG['Vmin']) * G * L[yc][:, None]
V_tr, V_te = volts(G_tr, y_tr['color']), volts(G_te, y_te['color'])
print(f'✓ dataset verified | train {nTR} (first {nTR} MNIST train images), test {nTE}')

# ---------------- reservoir (F1 reset, F2 C0 scaling) ----------------
NS = CFG['t_max'] // CFG['dt']; SPB = NS // CFG['nb']
def reset_vo2(c):
    v = c.VO2; T0 = torch.full((v.N,), CFG['T_base'] - 0.1, device=DEV)
    v.delta = torch.ones(v.N, device=DEV); v.reversed = torch.zeros(v.N, device=DEV)
    v.gr = v.g_major(T0); v.Tr = T0.clone(); v.Tpr = v.Tpr_func(); v.T_last = T0.clone()

def make_reservoir(c0, sigma, B):
    with torch.device(DEV):
        c = model.Circuit2D(B, 28, 28, 0.0, CFG['R0'], sigma, CFG['cth'], CFG['couple'], 1.0, CFG['T_base'])
    c.C0 = c.C0 * c0; c.R0C0 = c.R0 * c.C0                                     # F2
    assert abs(c.R0C0 - c.R0 * c.C0) < 1e-9 and abs(c.C0 - 145.34619293 * c0) < 1e-6
    return c

def simulate(c, V, seed):
    """Features = max of the VO2 current in each of 20 bins × 500 ns (layout neuron*20 + bin)."""
    B = V.shape[0]
    with torch.device(DEV):
        reset_vo2(c)                                                             # F1
        c.set_input(V=torch.as_tensor(V, dtype=torch.float32))
        y = torch.stack([torch.zeros(B, 784), torch.full((B, 784), float(CFG['T_base']))], dim=1)
        R_init = c.VO2.R(y[:, 1].reshape(-1)).reshape(B, 784)
        feat = torch.full((B, 784, CFG['nb']), -float('inf'))
        metal_prev = torch.zeros(B, 784, dtype=torch.bool)
        n_sw = torch.zeros(B, 784, dtype=torch.int32); t_first = torch.full((B, 784), -1.0)
        torch.manual_seed(seed)                                                  # same noise in every condition
        for i in range(NS):
            Rt = c.VO2.R(y[:, 1].reshape(-1)).reshape(B, 784)
            metal = Rt < 0.2 * R_init; onset = metal & ~metal_prev
            n_sw += onset.int(); t_first[onset & (t_first < 0)] = i * CFG['dt']; metal_prev = metal
            dy = c.step(0, y)
            b = i // SPB
            feat[:, :, b] = torch.maximum(feat[:, :, b], c.IR)
            y = y + dy * CFG['dt']
        assert torch.isfinite(feat).all(), 'non-finite features'
    return feat.reshape(B, -1).cpu().numpy(), n_sw.cpu().numpy(), t_first.cpu().numpy()

def extract(c0, sigma, V, offset, tag):
    n, B = len(V), CFG['batch']; X = np.zeros((n, 784 * CFG['nb']), np.float32)
    NSW = np.zeros((n, 784), np.int16); TF = np.zeros((n, 784), np.float32)
    c = make_reservoir(c0, sigma, B); t0 = time.time()
    for k, i0 in enumerate(range(0, n, B)):
        i1 = min(i0 + B, n); Vb = V[i0:i1]
        if i1 - i0 < B: Vb = np.vstack([Vb, np.full((B - (i1 - i0), 784), CFG['Vmin'], np.float32)])
        f, s, t = simulate(c, Vb, CFG['noise_seed'] * 10**7 + offset + k)
        X[i0:i1], NSW[i0:i1], TF[i0:i1] = f[:i1 - i0], s[:i1 - i0], t[:i1 - i0]
        if k == 0:
            est = (time.time() - t0) * n / B / 60
            print(f'    {tag}: first batch {time.time() - t0:.1f} s → ~{est:.0f} min for {n} images', flush=True)
    del c; torch.cuda.empty_cache() if DEV == 'cuda' else None
    return X, NSW, TF

# ---------------- P5 readout: standardised ridge, α on validation, refit on all train ----------------
TASKS = {'digit': 10, 'color': 10, 'parity': 2}
COLS, k_ = {}, 0
for t, m in TASKS.items(): COLS[t] = slice(k_, k_ + m); k_ += m
ALPHAS = 10.0 ** np.arange(-6, 6.01, 0.5)

def onehot(y):
    Y = -np.ones((len(y['digit']), k_)); r = np.arange(len(Y))
    Y[r, y['digit']] = 1; Y[r, 10 + y['color']] = 1; Y[r, 20 + y['parity']] = 1
    return Y

def readout(Xtr, Xte, idx=None):
    with torch.device('cpu'):
        if idx is not None: Xtr, Xte = Xtr[:, idx], Xte[:, idx]
        is_val = np.zeros(len(Xtr), bool)
        is_val[np.random.RandomState(CFG['split_seed']).permutation(len(Xtr))[:CFG['n_val']]] = True
        fit = ~is_val
        mu = Xtr[fit].mean(0, dtype=np.float64); sd = Xtr[fit].std(0, dtype=np.float64)
        const = sd < 1e-6; sd[const] = 1.0
        std = lambda X: torch.from_numpy(np.where(const, 0.0, (X - mu) / sd))       # float64
        Zf, Zv, Zt = std(Xtr[fit]), std(Xtr[is_val]), std(Xte)
        Y = torch.from_numpy(onehot(y_tr)); Yf, Yv = Y[torch.from_numpy(fit)], Y[torch.from_numpy(is_val)]
        nf = len(Zf); mf, yf = Zf.mean(0), Yf.mean(0)
        Gf = Zf.T @ Zf; Gv = Zv.T @ Zv; ZYf = Zf.T @ Yf; ZYv = Zv.T @ Yv
        lam, Q = torch.linalg.eigh(Gf - nf * torch.outer(mf, mf)); lam = lam.clamp(min=0)
        Pm = Q.T @ (ZYf - nf * torch.outer(mf, yf)); A = (Zv - mf) @ Q; del Q
        yv = {t: y_tr[t][is_val] for t in TASKS}; va = {t: [] for t in TASKS}
        for a in ALPHAS:
            S = A @ (Pm / (lam + a)[:, None]) + yf
            for t in TASKS: va[t].append(float((S[:, COLS[t]].argmax(1).numpy() == yv[t]).mean()))
        del A, Pm, lam
        best = {t: float(ALPHAS[int(np.argmax(va[t]))]) for t in TASKS}
        edge = [t for t in TASKS if best[t] in (ALPHAS[0], ALPHAS[-1])]
        n = len(Xtr); m = (Zf.sum(0) + Zv.sum(0)) / n; ym = Y.mean(0)
        Gc = Gf + Gv - n * torch.outer(m, m); ZYc = ZYf + ZYv - n * torch.outer(m, ym); del Gf, Gv
        Id = torch.eye(Gc.shape[0], dtype=torch.float64); pred = {}
        for t in TASKS:
            Lc, info = torch.linalg.cholesky_ex(Gc + best[t] * Id)
            W = torch.cholesky_solve(ZYc[:, COLS[t]], Lc) if int(info) == 0 else \
                torch.linalg.lstsq(Gc + best[t] * Id, ZYc[:, COLS[t]]).solution
            pred[t] = ((Zt - m) @ W + ym[COLS[t]]).argmax(1).numpy(); del Lc, W
        del Gc, ZYc, Id, Zf, Zv, Zt; gc.collect()
    acc = {t: float((pred[t] == y_te[t]).mean()) for t in TASKS}
    cm = np.zeros((10, 10), int); np.add.at(cm, (y_te['color'], pred['color']), 1)
    ix = {nm: i for i, nm in enumerate(NAMES)}; off = cm.sum() - np.trace(cm)
    inp = sum(cm[ix[a], ix[b]] + cm[ix[b], ix[a]] for a, b in [('Red', 'DkGreen'), ('Green', 'Orange')])
    return dict(test_acc=acc, alpha=best, alpha_at_grid_edge=edge, val_acc=va, n_masked=int(const.sum()),
                confusion_color=cm.tolist(), colour_err_in_pairs=float(inp / max(off, 1)),
                per_class_color={NAMES[i]: float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(10)})

REF = {'c0_1.0_sigma0': 'switches mean 3.366, first switch median 2790 ns (diagnostics/jitter, σ=0)',
       'c0_0.5_sigma0': 'switches mean 6.476 (diagnostics, σ=0)',
       'c0_1.0_sigma3e-4': 'original file: 79.83 / 56.88 / 85.39',
       'c0_0.5_sigma3e-4': 'original file: 89.72 / 78.06 / 92.23'}

# ---------------- main loop ----------------
for name, c0, sigma in RUN_SET:
    path = f'{OUT}/{name}.json'
    if os.path.exists(path): print(f'{name}: done — skip'); continue
    t0 = time.time(); print(f'\n===== {name}: C0 × {c0}, σ = {sigma:g} =====', flush=True)
    Xtr, _, _ = extract(c0, sigma, V_tr, 0, 'train')
    Xte, NSW, TF = extract(c0, sigma, V_te, 10**6, 'test')
    t_sim = (time.time() - t0) / 60
    ink = V_te > CFG['Vmin'] + 1e-6; ok = TF >= 0
    sw = dict(switches_mean=float(NSW.mean()), switches_median=float(np.median(NSW)),
              frac_ge3_switches=float((NSW >= 3).mean()), frac_never=float((~ok).mean()),
              t_first_median_ns=float(np.median(TF[ok])) if ok.any() else None,
              t_first_median_ink_ns=float(np.median(TF[ok & ink])) if (ok & ink).any() else None,
              t_first_median_bg_ns=float(np.median(TF[ok & ~ink])) if (ok & ~ink).any() else None,
              t_first_min_ns=float(TF[ok].min()) if ok.any() else None)
    feat_var = float(Xtr.var(0, dtype=np.float64).mean())
    print(f'    switching (test): {sw}\n    FeatVar = {feat_var:.4f}', flush=True)
    idx = None
    if SMOKE:   # small feature subset (bins 0–3) so the smoke test is quick
        idx = (np.arange(784)[:, None] * CFG['nb'] + np.arange(4)[None, :]).ravel()
    t1 = time.time(); r = readout(Xtr, Xte, idx); t_ro = (time.time() - t1) / 60
    a = r['test_acc']
    print(f'    {name}: digit {100*a["digit"]:.2f} | colour {100*a["color"]:.2f} | parity {100*a["parity"]:.2f}'
          f' | α {r["alpha"]}' + (f'  ⚠ α at grid edge: {r["alpha_at_grid_edge"]}' if r['alpha_at_grid_edge'] else ''))
    if name in REF: print(f'    reference: {REF[name]}')
    print(f'    simulation {t_sim:.1f} min, readout {t_ro:.1f} min', flush=True)
    res = dict(name=name, cond=dict(c0=c0, cth=CFG['cth'], noise=sigma), tau_el_pred_ns=1416.32309693658 * c0,
               protocol=dict(**CFG, alpha_grid=ALPHAS.tolist(), smoke=SMOKE,
                             split='first n_train MNIST train images; val = RandomState(split_seed).permutation[:n_val]',
                             readout='P5: standardised (fit stats), sd<1e-6 masked, ridge, α per task on val, refit on all train',
                             features='max VO2 current per 500-ns bin, 784×20' + (' (SMOKE: bins 0–3 only)' if SMOKE else '')),
               feat_var=feat_var, switching_test=sw, minutes=dict(sim=t_sim, readout=t_ro), **r)
    json.dump(res, open(path, 'w'), indent=1)
    del Xtr, Xte, NSW, TF; gc.collect()
    print(f'✓ saved {path}')

# ---------------- summary ----------------
print('\n========== SUMMARY (test accuracy, %) ==========')
for name, c0, sigma in RUN_SET:
    p = f'{OUT}/{name}.json'
    if os.path.exists(p):
        a = json.load(open(p))['test_acc']
        print(f'{name:18s} {100*a["digit"]:6.2f} {100*a["color"]:6.2f} {100*a["parity"]:6.2f}')
