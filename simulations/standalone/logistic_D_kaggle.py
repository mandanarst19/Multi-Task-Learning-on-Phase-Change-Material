# =============================================================================
# Baseline D (per-pixel sigmoids, no dynamics, no coupling) — ridge checks + LOGISTIC readout
# Paper 1, novelty 2 / 4. STANDALONE Kaggle cell. Needs paper1_data.json in the inputs, Internet ON.
# No model.py and no GPU needed (D has no simulation). CPU notebook is fine; RAM ~15 GB peak.
#
# What it does
#   1) builds D: 20 fixed sigmoids per pixel, thresholds = linspace(10.5, 12.2, 22)[1:-1],
#      width 1.7/40 V = 42.5 mV, feature = 1/(1+exp(-(V-θ)/w)), layout pixel*20 + k (15,680 features)
#   2) CHECK  ridge, 60k protocol (P5, val 10k)  → must reproduce 86.40 / 88.28 / 89.19 (2439 masked)
#   3) NEW    ridge, 20k protocol (P5, val 4k)   → like-for-like with the coupling runs
#                                                  (uncoupled reservoir 85.97 / 94.04 / 89.21)
#   4) NEW    logistic, 60k, colour (+ optional digit/parity), same settings as the reservoir logistic
#                                                  (reservoir: colour 90.45 %, raw input 37.87 %)
# Interpretation of 4: D + logistic colour  ≥ ~90 %  → reservoir and D are equal on colour with a
#                      nonlinear readout; < 90 %     → the reservoir features carry more colour information.
#
# HOW TO RUN: SMOKE = True first (~2 min, send me the output), then SMOKE = False + Save & Run All.
# Full run: ridge checks ~40 min, logistic colour ~2–3 h (3 values of C + refit).
# Send: /kaggle/working/logistic_D/logistic_D.json + the printed output.
# =============================================================================
import os, gc, json, time, numpy as np, torch
from torchvision import datasets
from sklearn.linear_model import LogisticRegression

SMOKE = True
LOG_TASKS = ['color']                 # add 'digit', 'parity' only if you want them (+ ~2 h each)
C_GRID = [1e-3, 1e-2, 1e-1]           # reservoir logistic picked C = 0.01 (lower edge of {0.01, 0.1, 1}); val was flat
MAX_ITER = 2000
CFG = dict(Vmin=10.5, Vmax=12.2, nb=20, split_seed=42)
P60 = dict(n_train=60_000, n_val=10_000, n_test=10_000)
P20 = dict(n_train=20_000, n_val=4_000, n_test=10_000)
CHUNK = 5000
if SMOKE:
    P60 = dict(n_train=3000, n_val=1000, n_test=1000); P20 = dict(n_train=2000, n_val=500, n_test=1000)
    C_GRID = [1e-2]; MAX_ITER = 100; CHUNK = 1000
OUT = '/kaggle/working/logistic_D' + ('_smoke' if SMOKE else ''); os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(os.cpu_count()); torch.set_grad_enabled(False)

def find(fname):
    for base in ('/kaggle/working', '/kaggle/input'):
        for r, _, fs in os.walk(base):
            for f in fs:
                if f == fname or (fname.endswith('.json') and f.startswith(fname[:-5]) and f.endswith('.json')):
                    return os.path.join(r, f)
    raise FileNotFoundError(fname)

# ---------------- dataset (verified reconstruction) ----------------
J = json.load(open(find('paper1_data.json')))
NAMES = J['sigma0_60k_final']['color_order']
assert NAMES == ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'Orange', 'Purple', 'DkGreen', 'Gray']
P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
              [1, .5, 0], [.5, 0, 1], [0, .5, 0], [.5, .5, .5]], np.float32)
L = P @ np.array([0.299, 0.587, 0.114], np.float32)
_lum = J['baselines_60k']['digit_acc_by_colour_sigma2e-4_seed0']
for _n, _l in zip(_lum['order_by_luminance'], _lum['luminance']):
    assert abs(L[NAMES.index(_n)] - _l) < 6e-4, _n
np.random.seed(42); YC_all = np.random.randint(0, 10, 60000); YC_te_all = np.random.randint(0, 10, 10000)
assert (np.bincount(YC_te_all, minlength=10) == np.array(J['sigma0_60k_final']['confusion_color']).sum(1)).all()
mn_tr = datasets.MNIST('/kaggle/working/data', train=True, download=True)
mn_te = datasets.MNIST('/kaggle/working/data', train=False, download=True)
G_all = mn_tr.data.numpy().reshape(60000, -1).astype(np.float32) / 255
G_te_all = mn_te.data.numpy().reshape(10000, -1).astype(np.float32) / 255
volts = lambda G, yc: CFG['Vmin'] + (CFG['Vmax'] - CFG['Vmin']) * G * L[yc][:, None]
V_all, V_te_all = volts(G_all, YC_all), volts(G_te_all, YC_te_all)
Y_all = {'digit': mn_tr.targets.numpy(), 'color': YC_all}; Y_te_all = {'digit': mn_te.targets.numpy(), 'color': YC_te_all}
for d in (Y_all, Y_te_all): d['parity'] = d['digit'] % 2
print('✓ dataset verified')

# ---------------- baseline D ----------------
THETA = np.linspace(CFG['Vmin'], CFG['Vmax'], CFG['nb'] + 2)[1:-1].astype(np.float32)
WIDTH = np.float32((CFG['Vmax'] - CFG['Vmin']) / 40)
assert len(THETA) == 20 and abs(WIDTH - 0.0425) < 1e-7
def D_features(V):
    """(n, 784) volts → (n, 15680) float32, layout pixel*20 + k"""
    X = np.empty((len(V), 784 * CFG['nb']), np.float32)
    for i0 in range(0, len(V), 5000):
        v = V[i0:i0 + 5000, :, None]
        X[i0:i0 + 5000] = (1.0 / (1.0 + np.exp(-(v - THETA) / WIDTH))).reshape(len(v), -1)
    return X

# ---------------- P5 ridge, chunked (same algebra as the reservoir readout, memory-light) ----------------
TASKS = {'digit': 10, 'color': 10, 'parity': 2}
COLS, k_ = {}, 0
for t, m in TASKS.items(): COLS[t] = slice(k_, k_ + m); k_ += m
ALPHAS = 10.0 ** np.arange(-6, 6.01, 0.5)

def onehot(y):
    Y = -np.ones((len(y['digit']), k_)); r = np.arange(len(Y))
    Y[r, y['digit']] = 1; Y[r, 10 + y['color']] = 1; Y[r, 20 + y['parity']] = 1
    return Y

def ridge_P5(Xtr, ytr, Xte, yte, n_val):
    t0 = time.time()
    is_val = np.zeros(len(Xtr), bool)
    is_val[np.random.RandomState(CFG['split_seed']).permutation(len(Xtr))[:n_val]] = True
    fit = ~is_val; fi, vi = np.where(fit)[0], np.where(is_val)[0]
    mu = Xtr[fit].mean(0, dtype=np.float64); sd = Xtr[fit].std(0, dtype=np.float64)
    const = sd < 1e-6; sd[const] = 1.0
    def Z(rows, X):
        z = X[rows].astype(np.float64); z -= mu; z /= sd; z[:, const] = 0.0
        return torch.from_numpy(z)
    Y = torch.from_numpy(onehot(ytr)); d = Xtr.shape[1]
    def gram(idx):
        G = torch.zeros(d, d, dtype=torch.float64); ZY = torch.zeros(d, k_, dtype=torch.float64)
        s = torch.zeros(d, dtype=torch.float64)
        for c0 in range(0, len(idx), CHUNK):
            r = idx[c0:c0 + CHUNK]; z = Z(r, Xtr); G += z.T @ z; ZY += z.T @ Y[r]; s += z.sum(0); del z
        return G, ZY, s
    Gf, ZYf, sf = gram(fi); Gv, ZYv, sv = gram(vi)
    nf = len(fi); mf, yf = sf / nf, Y[fi].mean(0)
    lam, Q = torch.linalg.eigh(Gf - nf * torch.outer(mf, mf)); lam = lam.clamp(min=0)
    Pm = Q.T @ (ZYf - nf * torch.outer(mf, yf)); A = (Z(vi, Xtr) - mf) @ Q; del Q; gc.collect()
    yv = {t: ytr[t][vi] for t in TASKS}; va = {t: [] for t in TASKS}
    for a in ALPHAS:
        S = A @ (Pm / (lam + a)[:, None]) + yf
        for t in TASKS: va[t].append(float((S[:, COLS[t]].argmax(1).numpy() == yv[t]).mean()))
    del A, Pm, lam
    best = {t: float(ALPHAS[int(np.argmax(va[t]))]) for t in TASKS}
    edge = [t for t in TASKS if best[t] in (ALPHAS[0], ALPHAS[-1])]
    n = len(Xtr); m = (sf + sv) / n; ym = Y.mean(0)
    Gc = Gf + Gv - n * torch.outer(m, m); ZYc = ZYf + ZYv - n * torch.outer(m, ym); del Gf, Gv; gc.collect()
    Id = torch.eye(d, dtype=torch.float64); pred = {}
    for t in TASKS:
        Lc, info = torch.linalg.cholesky_ex(Gc + best[t] * Id)
        W = torch.cholesky_solve(ZYc[:, COLS[t]], Lc) if int(info) == 0 else \
            torch.linalg.lstsq(Gc + best[t] * Id, ZYc[:, COLS[t]]).solution
        del Lc
        pr = []
        for c0 in range(0, len(Xte), CHUNK):
            pr.append(((Z(np.arange(c0, min(c0 + CHUNK, len(Xte))), Xte) - m) @ W + ym[COLS[t]]).argmax(1).numpy())
        pred[t] = np.concatenate(pr); del W
    del Gc, ZYc, Id; gc.collect()
    acc = {t: float((pred[t] == yte[t]).mean()) for t in TASKS}
    cm = np.zeros((10, 10), int); np.add.at(cm, (yte['color'], pred['color']), 1)
    ix = {nm: i for i, nm in enumerate(NAMES)}; off = cm.sum() - np.trace(cm)
    inp = sum(cm[ix[a], ix[b]] + cm[ix[b], ix[a]] for a, b in [('Red', 'DkGreen'), ('Green', 'Orange')])
    return dict(test_acc=acc, alpha=best, alpha_at_grid_edge=edge, val_acc=va, n_masked=int(const.sum()),
                confusion_color=cm.tolist(), colour_err_in_pairs=float(inp / max(off, 1)),
                per_class_color={NAMES[i]: float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(10)},
                minutes=(time.time() - t0) / 60)

def show(tag, r, ref=None):
    a = r['test_acc']
    print(f'{tag}: digit {100*a["digit"]:.2f} | colour {100*a["color"]:.2f} | parity {100*a["parity"]:.2f} | '
          f'masked {r["n_masked"]} | α {r["alpha"]} | {r["minutes"]:.1f} min'
          + (f'  ⚠ α at grid edge: {r["alpha_at_grid_edge"]}' if r['alpha_at_grid_edge'] else ''), flush=True)
    print('    per colour (%): ' + ', '.join(f'{k} {100*v:.1f}' for k, v in r['per_class_color'].items()))
    if ref: print(f'    reference: {ref}')

def subset(p):
    n = p['n_train']; nt = p['n_test']
    return (V_all[:n], {t: Y_all[t][:n] for t in Y_all}, V_te_all[:nt], {t: Y_te_all[t][:nt] for t in Y_te_all})

OUTJ = f'{OUT}/logistic_D.json'
RES = dict(smoke=SMOKE, definition='D: 20 sigmoids per pixel, θ = linspace(10.5, 12.2, 22)[1:-1], w = 1.7/40 V, '
                                   'x = 1/(1+exp(-(V-θ)/w)), layout pixel*20 + k',
           protocol=dict(P60=P60, P20=P20, C_grid=C_GRID, max_iter=MAX_ITER, log_tasks=LOG_TASKS))
save = lambda: json.dump(RES, open(OUTJ, 'w'), indent=1)

# ---------------- 2) CHECK: ridge 60k ----------------
Vtr, ytr, Vte, yte = subset(P60); Xtr, Xte = D_features(Vtr), D_features(Vte)
print(f'D features: {Xtr.shape}', flush=True)
RES['ridge_60k'] = r = ridge_P5(Xtr, ytr, Xte, yte, P60['n_val']); save()
show('RIDGE D 60k', r, 'baselines_60k.D: 86.40 / 88.28 / 89.19, masked 2439')
if not SMOKE:
    ok = all(abs(r['test_acc'][t] - v) < 6e-4 for t, v in (('digit', .864), ('color', .8828), ('parity', .8919)))
    print('    REPRODUCTION CHECK (D definition):', 'PASS' if ok else 'FAIL — the D formula differs; send me the output')
    RES['ridge_60k']['reproduction_pass'] = ok; save()

# ---------------- 3) NEW: ridge 20k (like-for-like with the coupling runs) ----------------
Vtr2, ytr2, Vte2, yte2 = subset(P20)
RES['ridge_20k'] = r2 = ridge_P5(D_features(Vtr2), ytr2, D_features(Vte2), yte2, P20['n_val']); save()
show('RIDGE D 20k', r2, 'reservoir 20k: uncoupled 85.97 / 94.04 / 89.21, coupled c = 0.02 90.19 / 78.99 / 92.56')
gc.collect()

# ---------------- 4) NEW: logistic 60k (same settings as the reservoir logistic) ----------------
perm = np.random.RandomState(CFG['split_seed']).permutation(len(Xtr))
is_val = np.zeros(len(Xtr), bool); is_val[perm[:P60['n_val']]] = True
order = np.r_[np.where(~is_val)[0], np.where(is_val)[0]]; nF = int((~is_val).sum())
Xa = Xtr[order]; del Xtr; gc.collect()
mu = Xa[:nF].mean(0, dtype=np.float64); sd = Xa[:nF].std(0, dtype=np.float64); keep = sd >= 1e-6
Xa = Xa[:, keep]; Xb = Xte[:, keep]; del Xte; gc.collect()
mu, sd = mu[keep].astype(np.float32), sd[keep].astype(np.float32); Xa -= mu; Xa /= sd; Xb -= mu; Xb /= sd
print(f'logistic: {Xa.shape[1]} features (dropped {int((~keep).sum())})', flush=True)
RES['logistic_60k'] = {}
for t in LOG_TASKS:
    ya, yb = ytr[t][order], yte[t]; val = {}; t1 = time.time()
    for Cc in C_GRID:
        t2 = time.time(); m = LogisticRegression(C=Cc, max_iter=MAX_ITER).fit(Xa[:nF], ya[:nF])
        val[Cc] = float((m.predict(Xa[nF:]) == ya[nF:]).mean())
        print(f'    D {t}: C={Cc:g} val {100*val[Cc]:.2f}% (iter {int(np.max(m.n_iter_))}/{MAX_ITER}, '
              f'{(time.time()-t2)/60:.1f} min)', flush=True)
        del m; gc.collect()
    best = max(val, key=val.get)
    m = LogisticRegression(C=best, max_iter=MAX_ITER).fit(Xa, ya); pred = m.predict(Xb)
    res = dict(C=best, C_at_grid_edge=best in (C_GRID[0], C_GRID[-1]), val_acc={str(k): v for k, v in val.items()},
               test_acc=float((pred == yb).mean()), n_iter=int(np.max(m.n_iter_)),
               converged=bool(np.max(m.n_iter_) < MAX_ITER), minutes=(time.time() - t1) / 60)
    if t == 'color':
        cm = np.zeros((10, 10), int); np.add.at(cm, (yb, pred), 1)
        ix = {nm: i for i, nm in enumerate(NAMES)}; off = cm.sum() - np.trace(cm)
        inp = sum(cm[ix[a_], ix[b_]] + cm[ix[b_], ix[a_]] for a_, b_ in [('Red', 'DkGreen'), ('Green', 'Orange')])
        res.update(confusion_color=cm.tolist(), colour_err_in_pairs=float(inp / max(off, 1)),
                   per_class_color={NAMES[i]: float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(10)})
    RES['logistic_60k'][t] = res; save(); del m; gc.collect()
    print(f'LOGISTIC D {t}: test {100*res["test_acc"]:.2f}% (C={best:g}, converged={res["converged"]}, '
          f'{res["minutes"]:.1f} min)' + ('  ⚠ C at grid edge' if res['C_at_grid_edge'] else ''), flush=True)
    if t == 'color':
        print('    per colour (%): ' + ', '.join(f'{k} {100*v:.1f}' for k, v in res['per_class_color'].items()))
        print(f'    pair-error share {100*res["colour_err_in_pairs"]:.1f}% | reference: reservoir + logistic 90.45 %, '
              f'raw input + logistic 37.87 %, D + ridge 88.28 %', flush=True)
print(f'✓ saved {OUTJ}')
