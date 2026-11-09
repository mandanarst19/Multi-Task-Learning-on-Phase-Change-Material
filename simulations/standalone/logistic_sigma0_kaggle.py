# =============================================================================
# Logistic readout on the σ = 0, 60k reservoir features — Paper 1, novelty 4 ("not the readout"?)
# STANDALONE Kaggle cell. Needs model.py + paper1_data.json in the inputs, Internet ON, GPU T4.
#
# What it does (C0 × 1.0, σ = 0, P5 protocol, 60k train / 10k val / 10k test):
#   1) rebuilds the 15,680 features of the final σ = 0 run (simulation on GPU, ~10 min);
#   2) ridge readout (P5)  → reproduction check: must give 91.38 / 81.48 / 93.43;
#   3) logistic (softmax) readout on the same standardised features, C chosen on validation,
#      for colour, digit and parity, with the colour confusion matrix and the pair-error share.
# Interpretation:
#   colour pairs stay near 50–60 % with logistic too → the limit is in the features (dynamics)
#   colour pairs rise clearly with logistic            → part of the limit is the linear readout
# HOW TO RUN: SMOKE = True first (~5 min, send me the output), then SMOKE = False + Save & Run All.
# =============================================================================
import os, sys, gc, json, time, importlib, numpy as np, torch
from torchvision import datasets

SMOKE = True
CFG = dict(dt=10, t_max=10_000, T_base=325, R0=12.0, cth=0.15, couple=0.02, Vmin=10.5, Vmax=12.2,
           nb=20, n_train=60_000, n_val=10_000, n_test=10_000, split_seed=42, noise_seed=0, batch=500)
C_GRID = [1e-2, 1e-1, 1.0]           # run 1: colour picked 1e-1 = upper edge of {1e-3, 1e-2, 1e-1}
MAX_ITER = 2000                      # run 1: 500 iterations did not converge
TASKS_LOG = ['color']                # digit logistic is known (94.47 %); add 'digit', 'parity' only if time allows
if SMOKE:
    CFG.update(n_train=600, n_val=200, n_test=300, batch=100); C_GRID = [1e-2]; MAX_ITER = 100
OUT = '/kaggle/working/logistic' + ('_smoke' if SMOKE else ''); os.makedirs(OUT, exist_ok=True)

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
        def std(X):                                   # float64, in place (memory-light at 60k)
            Z = X.astype(np.float64); Z -= mu; Z /= sd; Z[:, const] = 0.0
            return torch.from_numpy(Z)
        Zf, Zv, Zt = std(Xtr[fit]), std(Xtr[is_val]), std(Xte)
        Y = torch.from_numpy(onehot(y_tr)); Yf, Yv = Y[torch.from_numpy(fit)], Y[torch.from_numpy(is_val)]
        nf = len(Zf); mf, yf = Zf.mean(0), Yf.mean(0)
        Gf = Zf.T @ Zf; Gv = Zv.T @ Zv; ZYf = Zf.T @ Yf; ZYv = Zv.T @ Yv
        Zf_sum = Zf.sum(0); del Zf; gc.collect()
        lam, Q = torch.linalg.eigh(Gf - nf * torch.outer(mf, mf)); lam = lam.clamp(min=0)
        Pm = Q.T @ (ZYf - nf * torch.outer(mf, yf)); A = (Zv - mf) @ Q; del Q
        yv = {t: y_tr[t][is_val] for t in TASKS}; va = {t: [] for t in TASKS}
        for a in ALPHAS:
            S = A @ (Pm / (lam + a)[:, None]) + yf
            for t in TASKS: va[t].append(float((S[:, COLS[t]].argmax(1).numpy() == yv[t]).mean()))
        del A, Pm, lam
        best = {t: float(ALPHAS[int(np.argmax(va[t]))]) for t in TASKS}
        edge = [t for t in TASKS if best[t] in (ALPHAS[0], ALPHAS[-1])]
        n = len(Xtr); m = (Zf_sum + Zv.sum(0)) / n; ym = Y.mean(0)
        Gc = Gf + Gv - n * torch.outer(m, m); ZYc = ZYf + ZYv - n * torch.outer(m, ym); del Gf, Gv
        Id = torch.eye(Gc.shape[0], dtype=torch.float64); pred = {}
        for t in TASKS:
            Lc, info = torch.linalg.cholesky_ex(Gc + best[t] * Id)
            W = torch.cholesky_solve(ZYc[:, COLS[t]], Lc) if int(info) == 0 else \
                torch.linalg.lstsq(Gc + best[t] * Id, ZYc[:, COLS[t]]).solution
            pred[t] = ((Zt - m) @ W + ym[COLS[t]]).argmax(1).numpy(); del Lc, W
        del Gc, ZYc, Id, Zv, Zt; gc.collect()
    acc = {t: float((pred[t] == y_te[t]).mean()) for t in TASKS}
    cm = np.zeros((10, 10), int); np.add.at(cm, (y_te['color'], pred['color']), 1)
    ix = {nm: i for i, nm in enumerate(NAMES)}; off = cm.sum() - np.trace(cm)
    inp = sum(cm[ix[a], ix[b]] + cm[ix[b], ix[a]] for a, b in [('Red', 'DkGreen'), ('Green', 'Orange')])
    return dict(test_acc=acc, alpha=best, alpha_at_grid_edge=edge, val_acc=va, n_masked=int(const.sum()),
                confusion_color=cm.tolist(), colour_err_in_pairs=float(inp / max(off, 1)),
                per_class_color={NAMES[i]: float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(10)})


# ---------------- 1) features ----------------
t0 = time.time()
Xtr, _, _ = extract(1.0, 0.0, V_tr, 0, 'train')
Xte, NSW, TF = extract(1.0, 0.0, V_te, 10**6, 'test')
print(f'features: {Xtr.shape}, {Xte.shape} | FeatVar {Xtr.var(0, dtype=np.float64).mean():.4f} '
      f'(final run: 1.5826) | switches {NSW.mean():.3f} | {(time.time()-t0)/60:.1f} min', flush=True)
idx = (np.arange(784)[:, None] * CFG['nb'] + np.arange(4)[None, :]).ravel() if SMOKE else None

# ---------------- 2) ridge: reproduced in run 1 (91.38 / 81.47 / 93.43) — skipped here to save RAM ----------------
# ---------------- 3) logistic on the reservoir features AND on the raw input (control) ----------------
from sklearn.linear_model import LogisticRegression
if SMOKE: Xtr, Xte = Xtr[:, idx], Xte[:, idx]
perm = np.random.RandomState(CFG['split_seed']).permutation(len(Xtr))
is_val = np.zeros(len(Xtr), bool); is_val[perm[:CFG['n_val']]] = True
order = np.r_[np.where(~is_val)[0], np.where(is_val)[0]]          # fit rows first → slices, no copies
nF = int((~is_val).sum())

def standardise(Xa, Xb):
    """in-place float32 standardisation with fit-split statistics; near-constant columns dropped"""
    Xa = Xa[order]; mu = Xa[:nF].mean(0, dtype=np.float64); sd = Xa[:nF].std(0, dtype=np.float64)
    keep = sd >= 1e-6
    Xa = Xa[:, keep]; Xb = Xb[:, keep]; mu, sd = mu[keep].astype(np.float32), sd[keep].astype(np.float32)
    Xa -= mu; Xa /= sd; Xb -= mu; Xb /= sd
    return Xa, Xb, int((~keep).sum())

def run_logistic(Za, Zb, t, tag):
    ya, yb = y_tr[t][order], y_te[t]; val = {}; t1 = time.time()
    for Cc in C_GRID:
        t2 = time.time(); m = LogisticRegression(C=Cc, max_iter=MAX_ITER).fit(Za[:nF], ya[:nF])
        val[Cc] = float((m.predict(Za[nF:]) == ya[nF:]).mean())
        print(f'    {tag} {t}: C={Cc:g} val {100*val[Cc]:.2f}% (iter {int(np.max(m.n_iter_))}/{MAX_ITER}, '
              f'{(time.time()-t2)/60:.1f} min)', flush=True)
        del m; gc.collect()
    best = max(val, key=val.get)
    m = LogisticRegression(C=best, max_iter=MAX_ITER).fit(Za, ya); pred = m.predict(Zb)
    res = dict(C=best, C_at_grid_edge=best in (C_GRID[0], C_GRID[-1]), val_acc={str(k): v for k, v in val.items()},
               test_acc=float((pred == yb).mean()), n_iter=int(np.max(m.n_iter_)),
               converged=bool(np.max(m.n_iter_) < MAX_ITER), minutes=(time.time() - t1) / 60)
    if t == 'color':
        cm = np.zeros((10, 10), int); np.add.at(cm, (yb, pred), 1)
        ix = {nm: i for i, nm in enumerate(NAMES)}; off = cm.sum() - np.trace(cm)
        inp = sum(cm[ix[a_], ix[b_]] + cm[ix[b_], ix[a_]] for a_, b_ in [('Red', 'DkGreen'), ('Green', 'Orange')])
        res.update(confusion_color=cm.tolist(), colour_err_in_pairs=float(inp / max(off, 1)),
                   per_class_color={NAMES[i]: float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(10)})
    print(f'LOGISTIC {tag} {t}: test {100*res["test_acc"]:.2f}% (C={best:g}, converged={res["converged"]}, '
          f'{res["minutes"]:.1f} min)' + ('  ⚠ C at grid edge' if res['C_at_grid_edge'] else ''), flush=True)
    if t == 'color':
        print('    per class (%): ' + ', '.join(f'{k} {100*v:.1f}' for k, v in res['per_class_color'].items()))
        print(f'    pair-error share: {100*res["colour_err_in_pairs"]:.1f}%   '
              f'(ridge reservoir: Red 57.0, Green 58.2, Orange 56.0, DkGreen 54.3; share 82.2%)', flush=True)
    del m; gc.collect()
    return res

OUTJ = f'{OUT}/logistic_sigma0_60k_run2.json'
RES = dict(cond=dict(c0=1.0, cth=CFG['cth'], noise=0.0), smoke=SMOKE,
           protocol=dict(**CFG, C_grid=C_GRID, max_iter=MAX_ITER,
                         logistic='sklearn LogisticRegression (lbfgs, multinomial), float32 standardised (fit stats), near-constant dropped; C on val, refit on all train'),
           ridge_run1={'digit': 0.9138, 'color': 0.8147, 'parity': 0.9343},
           switching_test=dict(switches_mean=float(NSW.mean()), t_first_min_ns=float(TF[TF >= 0].min())),
           raw_input={}, reservoir={})
save = lambda: json.dump(RES, open(OUTJ, 'w'), indent=1)

# (a) control: logistic on the raw input voltages (784 features) — cheap, runs first
Ra, Rb, nm = standardise(V_tr.astype(np.float32), V_te.astype(np.float32))
print(f'raw input: {Ra.shape[1]} features (dropped {nm})', flush=True)
RES['raw_input']['color'] = run_logistic(Ra, Rb, 'color', 'RAW'); save(); del Ra, Rb; gc.collect()

# (b) reservoir features
Za, Zb, nm = standardise(Xtr, Xte); del Xtr, Xte; gc.collect()
print(f'reservoir: {Za.shape[1]} features (dropped {nm})', flush=True)
for t in TASKS_LOG:
    RES['reservoir'][t] = run_logistic(Za, Zb, t, 'RESERVOIR'); save()
print(f'✓ saved {OUTJ}')
