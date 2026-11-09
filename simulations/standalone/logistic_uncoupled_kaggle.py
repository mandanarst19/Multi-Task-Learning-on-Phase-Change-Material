# =============================================================================
# UNCOUPLED reservoir (σ = 0, 60k) — ridge + LOGISTIC colour readout — Paper 1, novelty 2
# STANDALONE Kaggle cell. Needs model.py + paper1_data.json (version with baseline_D_logistic_and_20k)
# in the inputs, Internet ON, GPU T4 (simulation), RAM ~15 GB peak.
#
# Question: with a NONLINEAR readout, do the uncoupled VO2 dynamics still encode colour better than the
# static per-pixel sigmoids D?   D + logistic (60k) = 92.19 % (C = 0.1, upper grid edge → lower bound)
#                                coupled reservoir + logistic (60k) = 90.45 %
# Uncoupled = neighbour heat flow removed (S_couple = 0), heat loss to the substrate kept at 0.92 Sth
# (identical definition to coupling0_kaggle.py; 20k ridge there: 85.97 / 94.04 / 89.21).
#
# Steps: 1) simulate 60k + 10k uncoupled features (~12 min GPU)
#        2) ridge P5, 60k (chunked)  → new like-for-like with D 60k (86.40 / 88.28 / 89.19) and coupled 60k
#                                      (91.38 / 81.47 / 93.43)
#        3) logistic colour, C grid {1e-3, 1e-2, 1e-1, 1} (wider than before so neither edge is hit)
#        4) optional (D_EXTRA = True): D + logistic at C = 1 to remove D's grid-edge caveat (+ ~1.5 h)
# HOW TO RUN: SMOKE = True first (~5 min, send me the output), then SMOKE = False + Save & Run All
# (≈ 4–5 h; ≈ 6 h with D_EXTRA). Send /kaggle/working/logistic_uncoupled/logistic_uncoupled.json + output.
# =============================================================================
import os, sys, gc, json, time, importlib, numpy as np, torch
from torchvision import datasets
from sklearn.linear_model import LogisticRegression

SMOKE = True
D_EXTRA = False
CFG = dict(dt=10, t_max=10_000, T_base=325, R0=12.0, cth=0.15, couple=0.02, Vmin=10.5, Vmax=12.2,
           nb=20, n_train=60_000, n_val=10_000, n_test=10_000, split_seed=42, noise_seed=0, batch=500)
C_GRID = [1e-3, 1e-2, 1e-1, 1.0]
MAX_ITER = 2000
CHUNK = 5000
if SMOKE:
    CFG.update(n_train=1500, n_val=500, n_test=500, batch=100); C_GRID = [1e-2]; MAX_ITER = 100; CHUNK = 500
OUT = '/kaggle/working/logistic_uncoupled' + ('_smoke' if SMOKE else ''); os.makedirs(OUT, exist_ok=True)
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
    c.S_couple = 0.0                                                           # UNCOUPLED: neighbour term removed
    assert abs(c.R0C0 - c.R0 * c.C0) < 1e-9 and abs(c.C0 - 145.34619293 * c0) < 1e-6
    assert abs(c.S_env - 0.92 * c.Sth) < 1e-12 and c.S_couple == 0.0         # substrate loss unchanged
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


assert 'baseline_D_logistic_and_20k' in J, 'upload the paper1_data.json that has baseline_D_logistic_and_20k'
Y_te = y_te
OUTJ = f'{OUT}/logistic_uncoupled.json'
RES = dict(smoke=SMOKE, cond=dict(c0=1.0, cth=CFG['cth'], noise=0.0, S_couple_over_Sth=0.0, S_env_over_Sth=0.92),
           protocol=dict(**CFG, C_grid=C_GRID, max_iter=MAX_ITER,
                         logistic='sklearn LogisticRegression (lbfgs, multinomial), float32 standardised (fit stats), '
                                  'near-constant dropped; C on val, refit on all train (identical to the D / coupled runs)'),
           references=dict(D_ridge_60k=[86.40, 88.28, 89.19], coupled_ridge_60k=[91.38, 81.47, 93.43],
                           D_logistic_colour_60k=92.19, coupled_logistic_colour_60k=90.45,
                           uncoupled_ridge_20k=[85.97, 94.04, 89.21]))
save = lambda: json.dump(RES, open(OUTJ, 'w'), indent=1)

# ---------------- 1) uncoupled features ----------------
t0 = time.time()
Xtr, _, _ = extract(1.0, 0.0, V_tr, 0, 'train')
Xte, NSW, TF = extract(1.0, 0.0, V_te, 10**6, 'test')
ink = V_te > CFG['Vmin'] + 1e-6; ok = TF >= 0
RES['switching_test'] = dict(switches_mean=float(NSW.mean()), t_first_median_bg_ns=float(np.median(TF[ok & ~ink])),
                             t_first_min_ns=float(TF[ok].min()))
print(f'features {Xtr.shape} | switches {NSW.mean():.3f} (20k uncoupled run: 3.113) | background first switch '
      f'{RES["switching_test"]["t_first_median_bg_ns"]:.0f} ns (expected 2840) | {(time.time()-t0)/60:.1f} min', flush=True)
del NSW, TF; gc.collect()

# ---------------- 2) ridge P5, 60k ----------------
RES['ridge_60k'] = r = ridge_P5(Xtr, y_tr, Xte, y_te, CFG['n_val']); save()
show('RIDGE uncoupled 60k', r, 'D 60k 86.40 / 88.28 / 89.19 | coupled 60k 91.38 / 81.47 / 93.43 | uncoupled 20k 85.97 / 94.04 / 89.21')

# ---------------- 3) logistic colour ----------------
perm = np.random.RandomState(CFG['split_seed']).permutation(len(Xtr))
is_val = np.zeros(len(Xtr), bool); is_val[perm[:CFG['n_val']]] = True
order = np.r_[np.where(~is_val)[0], np.where(is_val)[0]]; nF = int((~is_val).sum())

def prep(Xa, Xb):
    Xa = Xa[order]; mu = Xa[:nF].mean(0, dtype=np.float64); sd = Xa[:nF].std(0, dtype=np.float64); keep = sd >= 1e-6
    Xa = Xa[:, keep]; Xb = Xb[:, keep]; mu, sd = mu[keep].astype(np.float32), sd[keep].astype(np.float32)
    Xa -= mu; Xa /= sd; Xb -= mu; Xb /= sd
    return Xa, Xb, int((~keep).sum())

def logistic(Xa, Xb, grid, tag):
    ya, yb = y_tr['color'][order], y_te['color']; val = {}; t1 = time.time()
    for Cc in grid:
        t2 = time.time(); m = LogisticRegression(C=Cc, max_iter=MAX_ITER).fit(Xa[:nF], ya[:nF])
        val[Cc] = float((m.predict(Xa[nF:]) == ya[nF:]).mean())
        print(f'    {tag} colour: C={Cc:g} val {100*val[Cc]:.2f}% (iter {int(np.max(m.n_iter_))}/{MAX_ITER}, '
              f'{(time.time()-t2)/60:.1f} min)', flush=True)
        del m; gc.collect()
    best = max(val, key=val.get)
    m = LogisticRegression(C=best, max_iter=MAX_ITER).fit(Xa, ya); pred = m.predict(Xb)
    cm = np.zeros((10, 10), int); np.add.at(cm, (yb, pred), 1)
    ix = {nm: i for i, nm in enumerate(NAMES)}; off = cm.sum() - np.trace(cm)
    inp = sum(cm[ix[a_], ix[b_]] + cm[ix[b_], ix[a_]] for a_, b_ in [('Red', 'DkGreen'), ('Green', 'Orange')])
    res = dict(C=best, C_at_grid_edge=best in (grid[0], grid[-1]), val_acc={str(k): v for k, v in val.items()},
               test_acc=float((pred == yb).mean()), n_iter=int(np.max(m.n_iter_)),
               converged=bool(np.max(m.n_iter_) < MAX_ITER), minutes=(time.time() - t1) / 60,
               confusion_color=cm.tolist(), colour_err_in_pairs=float(inp / max(off, 1)),
               per_class_color={NAMES[i]: float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(10)})
    del m; gc.collect()
    print(f'LOGISTIC {tag} colour: test {100*res["test_acc"]:.2f}% (C={best:g}, converged={res["converged"]}, '
          f'{res["minutes"]:.1f} min)' + ('  ⚠ C at grid edge' if res['C_at_grid_edge'] else ''), flush=True)
    print('    per colour (%): ' + ', '.join(f'{k} {100*v:.1f}' for k, v in res['per_class_color'].items()))
    print(f'    pair-error share {100*res["colour_err_in_pairs"]:.1f}%', flush=True)
    return res

Xa, Xb, nd = prep(Xtr, Xte); del Xtr, Xte; gc.collect()
print(f'logistic (uncoupled): {Xa.shape[1]} features (dropped {nd})', flush=True)
RES['logistic_60k_colour'] = lr = logistic(Xa, Xb, C_GRID, 'UNCOUPLED'); save(); del Xa, Xb; gc.collect()
print(f'    reference: D + logistic ≥ 92.19 %, coupled + logistic 90.45 %, raw input + logistic 37.87 %')
print(f'    VERDICT: uncoupled {"ABOVE" if lr["test_acc"] > 0.9219 else "NOT above"} D + logistic (92.19 %)', flush=True)

# ---------------- 4) optional: D at C = 1 (removes D's grid-edge caveat) ----------------
if D_EXTRA:
    Xa, Xb, nd = prep(D_features(V_tr), D_features(V_te))
    print(f'logistic (D): {Xa.shape[1]} features (dropped {nd})', flush=True)
    RES['D_logistic_60k_colour_C1'] = logistic(Xa, Xb, [1.0], 'D (C = 1 only; val at C = 0.1 was 90.73 %)'); save()
    del Xa, Xb; gc.collect()
print(f'✓ saved {OUTJ}')
