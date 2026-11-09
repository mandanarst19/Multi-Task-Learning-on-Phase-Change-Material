# =============================================================================
# R1 — FINAL σ = 0 RUN, 60k, ALL TIME WINDOWS  (RECONSTRUCTION of the lost notebook)
# Paper 1 → paper1_data.json: sigma0_60k_final (Table II, sliding windows, confusions)
# =============================================================================
# RECONSTRUCTED, NOT THE ORIGINAL: the original notebook is lost. This script uses the verified core
# (it already reproduced 91.38 / 81.47 / 93.43) and the readout logic of the T1 notebook (one Gram,
# windows by sub-indexing, eigh ≤ 6000 features, per-α Cholesky above). Every window is compared
# with paper1_data.json at the end (tolerance 0.6 pp): PASS = this code is a valid replacement.
# Inputs: model.py + paper1_data.json. Internet ON. GPU T4. Full run ≈ 1.5 h simulation + ~20 min readout.
# HOW TO RUN: SMOKE = True interactively (~5 min) → send me the output; then SMOKE = False + Save & Run All.
# Outputs (/kaggle/working/final_sigma0/): result_sigma0_60k.json, X_train.npy / X_test.npy (if KEEP_FEATURES),
#          yd/yc/yp_tr.npy and _te.npy (labels, for later readouts).
# =============================================================================
SMOKE = True
KEEP_FEATURES = True          # 3.8 GB + 0.6 GB in /kaggle/working; needed for any later readout
CFG = dict(dt=10, t_max=10_000, T_base=325, R0=12.0, cth=0.15, couple=0.02, Vmin=10.5, Vmax=12.2,
           nb=20, n_train=60_000, n_val=10_000, n_test=10_000, split_seed=42, noise_seed=0, batch=500)
if SMOKE: CFG.update(n_train=600, n_val=200, n_test=300, batch=100)
OUT = '/kaggle/working/final_sigma0' + ('_smoke' if SMOKE else ''); import os; os.makedirs(OUT, exist_ok=True)
# ======================= SHARED CORE (identical in every reconstructed script) =======================
# Source: paper1/kaggle/c0_sigma_grid_kaggle.py + logistic_sigma0_kaggle.py (verified: the same core
# reproduced the final σ = 0 run, ridge 91.38 / 81.47 / 93.43). Fixes F1 (reset), F2 (R0C0), F7 (reversal).
import os, sys, gc, json, time, types, importlib, numpy as np, torch
from torchvision import datasets

def find(fname):
    for base in ('/kaggle/working', '/kaggle/input'):
        for r, _, fs in os.walk(base):
            for f in fs:
                if f == fname or (fname.endswith('.json') and f.startswith(fname[:-5]) and f.endswith('.json')):
                    return os.path.join(r, f)
    raise FileNotFoundError(fname)
sys.path.insert(0, os.path.dirname(find('model.py'))); import model; importlib.reload(model)
assert model.VO2.reversal.__name__ == 'reversal'                                   # F7
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False); torch.set_num_threads(os.cpu_count())
print('device for simulation:', DEV, '| CPU threads:', os.cpu_count())

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
YD_all, YD_te_all = mn_tr.targets.numpy(), mn_te.targets.numpy()
volts = lambda G, yc: (CFG['Vmin'] + (CFG['Vmax'] - CFG['Vmin']) * G * L[yc][:, None]).astype(np.float32)
def labels(yd, yc): return {'digit': yd, 'color': yc, 'parity': yd % 2}
print('✓ dataset verified')

# ---------------- reservoir ----------------
def reset_vo2(c):                                                                   # F1
    v = c.VO2; T0 = torch.full((v.N,), CFG['T_base'] - 0.1, device=DEV)
    v.delta = torch.ones(v.N, device=DEV); v.reversed = torch.zeros(v.N, device=DEV)
    v.gr = v.g_major(T0); v.Tr = T0.clone(); v.Tpr = v.Tpr_func(); v.T_last = T0.clone()

def make_reservoir(c0, sigma, B):
    with torch.device(DEV):
        c = model.Circuit2D(B, 28, 28, 0.0, CFG['R0'], sigma, CFG['cth'], CFG['couple'], 1.0, CFG['T_base'])
    c.C0 = c.C0 * c0; c.R0C0 = c.R0 * c.C0                                          # F2
    assert abs(c.C0 - 145.34619293 * c0) < 1e-6
    return c

def simulate(c, V, seed, dt=None, trace=False, count_rev=False):
    """features = max VO2 current per 500-ns bin (layout neuron*20 + bin);
    switch = R < 0.2·R(t=0); returns dict of numpy arrays."""
    dt = dt or CFG['dt']; NS = CFG['t_max'] // dt; SPB = NS // CFG['nb']; B = V.shape[0]
    with torch.device(DEV):
        reset_vo2(c)
        if count_rev: c.VO2.rev_count = torch.zeros(c.VO2.N)
        c.set_input(V=torch.as_tensor(V, dtype=torch.float32))
        y = torch.stack([torch.zeros(B, 784), torch.full((B, 784), float(CFG['T_base']))], dim=1)
        R_init = c.VO2.R(y[:, 1].reshape(-1)).reshape(B, 784)
        feat = torch.full((B, 784, CFG['nb']), -float('inf'))
        metal_prev = torch.zeros(B, 784, dtype=torch.bool)
        n_sw = torch.zeros(B, 784, dtype=torch.int32); t_sw = torch.full((B, 784, 3), -1.0)
        onset_t = torch.zeros(NS); I_tr = torch.zeros(NS, B, 784, dtype=torch.float16) if trace else None
        torch.manual_seed(seed)
        for i in range(NS):
            Rt = c.VO2.R(y[:, 1].reshape(-1)).reshape(B, 784)
            metal = Rt < 0.2 * R_init; onset = metal & ~metal_prev; metal_prev = metal
            for k in range(3):                                   # times of the 1st, 2nd, 3rd switch
                sel = onset & (n_sw == k); t_sw[..., k][sel] = i * dt
            n_sw += onset.int(); onset_t[i] = onset.sum()
            dy = c.step(0, y)
            feat[:, :, i // SPB] = torch.maximum(feat[:, :, i // SPB], c.IR)
            if trace: I_tr[i] = c.IR.half()
            y = y + dy * dt
        assert torch.isfinite(feat).all(), 'non-finite features'
        out = dict(X=feat.reshape(B, -1).cpu().numpy(), n_sw=n_sw.cpu().numpy(), t_sw=t_sw.cpu().numpy(),
                   onset_t=onset_t.cpu().numpy())
        if trace: out['I'] = I_tr.cpu().numpy()
        if count_rev: out['rev'] = c.VO2.rev_count.reshape(B, 784).cpu().numpy()
    return out

def reversal_counting(self, T):
    """Copy of model.VO2.reversal that also counts reversals per neuron (diagnostics only).
    Attached to ONE instance with types.MethodType — the class method stays the original (F7)."""
    T = T.clamp(305, 370)
    dT = T - self.T_last
    if dT.abs().max() > 0.01:
        delta = torch.sign(dT)
        reversal_mask = (delta != self.delta) & (delta != 0)
        if reversal_mask.any():
            self.gr[reversal_mask] = self.g(T)[reversal_mask]
            self.delta[reversal_mask] = delta[reversal_mask]
            self.reversed[reversal_mask] = 1
            self.Tr[reversal_mask] = T[reversal_mask]
            self.Tpr[reversal_mask] = self.Tpr_func()[reversal_mask]
            self.rev_count += reversal_mask.float()
        self.T_last = T

def extract(c0, sigma, V, offset, tag, noise_seed=0, path=None):
    """features for all rows of V (padded last batch); optional memmap at `path` (resumable)."""
    n, B = len(V), CFG['batch']; nb = (n + B - 1) // B
    if path:
        prog = path + '.progress'
        X = np.lib.format.open_memmap(path, mode='r+' if os.path.exists(path) else 'w+', dtype=np.float32,
                                      shape=(n, 784 * CFG['nb']))
        start = int(open(prog).read()) if os.path.exists(prog) else 0
    else:
        X = np.zeros((n, 784 * CFG['nb']), np.float32); start = 0
    NSW = np.zeros((n, 784), np.int16); TF = np.full((n, 784), -1.0, np.float32)
    if start < nb:
        c = make_reservoir(c0, sigma, B); t0 = time.time()
        for k in range(start, nb):
            i0, i1 = k * B, min((k + 1) * B, n); Vb = V[i0:i1]
            if i1 - i0 < B: Vb = np.vstack([Vb, np.full((B - (i1 - i0), 784), CFG['Vmin'], np.float32)])
            o = simulate(c, Vb, noise_seed * 10**7 + offset + k)
            X[i0:i1] = o['X'][:i1 - i0]; NSW[i0:i1] = o['n_sw'][:i1 - i0]; TF[i0:i1] = o['t_sw'][:i1 - i0, :, 0]
            if k == start:
                print(f'    {tag}: first batch {time.time()-t0:.1f} s → ~{(time.time()-t0)*(nb-start)/60:.0f} min', flush=True)
            if path and (k % 20 == 0 or k == nb - 1):
                X.flush(); open(prog, 'w').write(str(k + 1))
        del c; torch.cuda.empty_cache() if DEV == 'cuda' else None
    return X, NSW, TF

# ---------------- P5 ridge (one Gram, any feature subset) ----------------
TASKS = {'digit': 10, 'color': 10, 'parity': 2}
COLS, K_ = {}, 0
for t, m in TASKS.items(): COLS[t] = slice(K_, K_ + m); K_ += m
ALPHAS = 10.0 ** np.arange(-6, 6.01, 0.5)
RDEV = DEV                                   # Gram/solves in float64 on the GPU (T4: ~16 GB)

def onehot(y):
    Y = -np.ones((len(y['digit']), K_)); r = np.arange(len(Y))
    Y[r, y['digit']] = 1; Y[r, 10 + y['color']] = 1; Y[r, 20 + y['parity']] = 1
    return Y

class Ridge:
    """Standardise with fit-split stats (sd < 1e-6 masked), accumulate fit/val Grams once,
    then solve any window by sub-indexing. α per task on validation, refit on fit+val."""
    def __init__(self, Xtr, ytr, Xte, n_val, chunk=2000):
        n, d = Xtr.shape; f64 = torch.float64
        self.is_val = np.zeros(n, bool)
        self.is_val[np.random.RandomState(CFG['split_seed']).permutation(n)[:n_val]] = True
        s = np.zeros(d); ss = np.zeros(d)
        for i in range(0, n, chunk):
            x = np.asarray(Xtr[i:i + chunk], np.float64)[~self.is_val[i:i + chunk]]; s += x.sum(0); ss += (x ** 2).sum(0)
        nf = int((~self.is_val).sum()); mu = s / nf; sd = np.sqrt(np.maximum(ss / nf - mu ** 2, 0))
        self.const = sd < 1e-6; sd[self.const] = 1.0
        mu_t, sd_t = torch.tensor(mu, dtype=f64, device=RDEV), torch.tensor(sd, dtype=f64, device=RDEV)
        keep = torch.tensor(~self.const, dtype=f64, device=RDEV)
        self.stdz = lambda x: ((torch.tensor(np.asarray(x, np.float64), dtype=f64, device=RDEV) - mu_t) / sd_t) * keep
        Y = onehot(ytr); self.ytr = ytr
        self.G = {p: torch.zeros(d, d, dtype=f64, device=RDEV) for p in ('fit', 'val')}
        self.S = {p: torch.zeros(d, dtype=f64, device=RDEV) for p in ('fit', 'val')}
        self.ZY = {p: torch.zeros(d, K_, dtype=f64, device=RDEV) for p in ('fit', 'val')}
        self.SY = {p: torch.zeros(K_, dtype=f64, device=RDEV) for p in ('fit', 'val')}
        self.n = {'fit': 0, 'val': 0}
        self.Zval = np.zeros((n_val, d), np.float32); vp = 0
        for i in range(0, n, chunk):
            z = self.stdz(Xtr[i:i + chunk]); yb = torch.tensor(Y[i:i + chunk], dtype=f64, device=RDEV)
            mv = torch.tensor(self.is_val[i:i + chunk], device=RDEV)
            for p, m in (('fit', ~mv), ('val', mv)):
                zp, yp = z[m], yb[m]
                self.G[p] += zp.T @ zp; self.S[p] += zp.sum(0); self.ZY[p] += zp.T @ yp
                self.SY[p] += yp.sum(0); self.n[p] += int(m.sum())
            zv = z[mv].float().cpu().numpy(); self.Zval[vp:vp + len(zv)] = zv; vp += len(zv); del z
        self.Zte = np.concatenate([self.stdz(Xte[i:i + chunk]).float().cpu().numpy() for i in range(0, len(Xte), chunk)])
        self.yval = {t: ytr[t][self.is_val] for t in TASKS}

    @staticmethod
    def _solve(M, a, B_):
        M = M.clone(); M.diagonal().add_(a)
        L_, info = torch.linalg.cholesky_ex(M)
        if int(info) == 0: X_ = torch.cholesky_solve(B_, L_)
        else: X_ = torch.linalg.solve(M, B_)
        del M, L_; return X_

    def window(self, idx):
        f64 = torch.float64; it = torch.tensor(idx, device=RDEV); nf = self.n['fit']
        full = len(idx) == self.G['fit'].shape[0]
        sub = lambda G: G.clone() if full else G[it[:, None], it]           # in-place below: memory-safe at 15,680
        mf = self.S['fit'][it] / nf; yf = self.SY['fit'] / nf
        Gc = sub(self.G['fit']); Gc.addr_(mf, mf, alpha=-nf)
        ZYc = self.ZY['fit'][it] - nf * torch.outer(mf, yf)
        Zv = torch.tensor(self.Zval[:, idx], dtype=f64, device=RDEV) - mf
        va = {t: [] for t in TASKS}
        if len(idx) <= 6000:                                    # eigh once, all α
            lam, Q = torch.linalg.eigh(Gc); lam = lam.clamp(min=0); Pm = Q.T @ ZYc; A = Zv @ Q; del Q
            for a in ALPHAS:
                Sc = A @ (Pm / (lam + a)[:, None]) + yf
                for t in TASKS: va[t].append(float((Sc[:, COLS[t]].argmax(1).cpu().numpy() == self.yval[t]).mean()))
            del A, Pm, lam
        else:                                                   # memory-safe: one Cholesky per α
            for a in ALPHAS:
                Sc = Zv @ self._solve(Gc, a, ZYc) + yf
                for t in TASKS: va[t].append(float((Sc[:, COLS[t]].argmax(1).cpu().numpy() == self.yval[t]).mean()))
        del Gc, ZYc, Zv
        best = {t: float(ALPHAS[int(np.argmax(va[t]))]) for t in TASKS}
        edge = [t for t in TASKS if best[t] in (ALPHAS[0], ALPHAS[-1])]
        nt = nf + self.n['val']; m_ = (self.S['fit'][it] + self.S['val'][it]) / nt; y_ = (self.SY['fit'] + self.SY['val']) / nt
        Gc = sub(self.G['fit']); Gc += sub(self.G['val']); Gc.addr_(m_, m_, alpha=-nt)
        ZYc = self.ZY['fit'][it] + self.ZY['val'][it] - nt * torch.outer(m_, y_)
        Zt = torch.tensor(self.Zte[:, idx], dtype=f64, device=RDEV) - m_; pred = {}
        for t in TASKS:
            W = self._solve(Gc, best[t], ZYc[:, COLS[t]]); pred[t] = (Zt @ W + y_[COLS[t]]).argmax(1).cpu().numpy(); del W
        del Gc, ZYc, Zt; torch.cuda.empty_cache() if DEV == 'cuda' else None
        return dict(alpha=best, alpha_at_grid_edge=edge, val_acc_curve=va), pred

def win_idx(b0, b1): return (np.arange(784)[:, None] * CFG['nb'] + np.arange(b0, b1)[None, :]).ravel()

def colour_stats(yc, pc):
    cm = np.zeros((10, 10), int); np.add.at(cm, (yc, pc), 1)
    ix = {nm: i for i, nm in enumerate(NAMES)}; off = cm.sum() - np.trace(cm)
    pairs = sum(cm[ix[a], ix[b]] + cm[ix[b], ix[a]] for a, b in [('Red', 'DkGreen'), ('Green', 'Orange')])
    triad = pairs + sum(cm[ix[a], ix[b]] + cm[ix[b], ix[a]] for a, b in [('Red', 'Purple'), ('DkGreen', 'Purple')])
    return dict(confusion_color=cm.tolist(), colour_err_in_pairs=float(pairs / max(off, 1)),
                colour_err_in_pairs_plus_purple=float(triad / max(off, 1)),
                per_class_color={NAMES[i]: float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(10)})

def check(tag, got, ref, tol=0.006):
    ok = all(abs(got[t] - ref[t]) <= tol for t in TASKS)
    print(f'    REPRODUCTION {tag}: got ' + ' / '.join(f'{100*got[t]:.2f}' for t in TASKS) +
          ' | ref ' + ' / '.join(f'{100*ref[t]:.2f}' for t in TASKS) + f' → {"PASS" if ok else "DIFFERENT — send me the output"}', flush=True)
    return ok
# ===================== end of shared core =====================

# ---------------- data ----------------
nTR, nTE = CFG['n_train'], CFG['n_test']
V_tr, V_te = volts(G_all[:nTR], YC_all[:nTR]), volts(G_te_all[:nTE], YC_te_all[:nTE])
y_tr, y_te = labels(YD_all[:nTR], YC_all[:nTR]), labels(YD_te_all[:nTE], YC_te_all[:nTE])
for sp, y in (('tr', y_tr), ('te', y_te)):
    for k, t in (('yd', 'digit'), ('yc', 'color'), ('yp', 'parity')): np.save(f'{OUT}/{k}_{sp}.npy', y[t])

# ---------------- 1) features (resumable memmaps) ----------------
t0 = time.time()
Xtr, _, _ = extract(1.0, 0.0, V_tr, 0, 'train', path=f'{OUT}/X_train.npy')
Xte, NSW, TF = extract(1.0, 0.0, V_te, 10**6, 'test', path=f'{OUT}/X_test.npy')
t_sim = (time.time() - t0) / 60
ok = TF >= 0
print(f'features {Xtr.shape} {Xte.shape} | FeatVar {np.asarray(Xtr[:5000]).var(0, dtype=np.float64).mean():.4f} '
      f'(first 5000 rows; final run, all rows: 1.5826) | earliest switch {TF[ok].min() if ok.any() else None} ns '
      f'(final: 2030) | switches mean {NSW.mean():.3f} | {t_sim:.1f} min', flush=True)

# ---------------- 2) readout ----------------
WINDOWS = {'full (0-19)': (0, 20), 'bins 0-3': (0, 4), 'early 0-7': (0, 8), 'bins 4-19': (4, 20),
           'steady 8-19': (8, 20), 'late 14-19': (14, 20), 'cum 0-1us': (0, 2), 'cum 0-3us': (0, 6),
           'cum 0-6us': (0, 12)}                       # 'cum 0-4us' == 'early 0-7' (same bins) — kept once
if SMOKE: WINDOWS = {k: WINDOWS[k] for k in ('full (0-19)', 'bins 0-3', 'cum 0-1us')}
t1 = time.time(); R = Ridge(Xtr, y_tr, Xte, CFG['n_val'])
print(f'Gram built ({(time.time()-t1)/60:.1f} min) | masked features: {int(R.const.sum())} (final: 263)', flush=True)
REF = J['sigma0_60k_final']['windows']
res = dict(sigma=0.0, n_train=nTR, n_val=CFG['n_val'], n_test=nTE, n_masked=int(R.const.sum()),
           simulation_minutes=t_sim, alpha_grid=ALPHAS.tolist(), smoke=SMOKE, reconstructed=True,
           protocol=dict(**CFG, readout='P5 ridge; α per task on val; refit on fit+val; eigh ≤ 6000 features, per-α Cholesky above'),
           windows={}, sliding={}, checks={})
for wn, (b0, b1) in WINDOWS.items():
    r, pred = R.window(win_idx(b0, b1)); acc = {t: float((pred[t] == y_te[t]).mean()) for t in TASKS}
    r['test_acc'] = acc; r['bins_half_open'] = [b0, b1]; r['time_us'] = [b0 * 0.5, b1 * 0.5]
    res['windows'][wn] = r
    print(f'  {wn:12s}: ' + ' | '.join(f'{t}={100*acc[t]:.2f}%' for t in TASKS) + f'  α={r["alpha"]}'
          + (f'  ⚠ edge:{r["alpha_at_grid_edge"]}' if r['alpha_at_grid_edge'] else ''), flush=True)
    if wn.startswith('full'):
        res.update(colour_stats(y_te['color'], pred['color']))
        for t, k in (('digit', 10), ('parity', 2)):
            cm = np.zeros((k, k), int); np.add.at(cm, (y_te[t], pred[t]), 1); res[f'confusion_{t}'] = cm.tolist()
        cmd = np.array(res['confusion_digit'])
        res['per_class_digit'] = {str(i): float(cmd[i, i] / max(cmd[i].sum(), 1)) for i in range(10)}
        res['digit_acc_by_colour'] = {NAMES[c]: float((pred['digit'][y_te['color'] == c] == y_te['digit'][y_te['color'] == c]).mean())
                                      for c in range(10)}
    if not SMOKE: res['checks'][wn] = check(wn, acc, REF[wn]['test_acc'])
for b in range(CFG['nb'] - 1):
    _, pred = R.window(win_idx(b, b + 2)); res['sliding'][str(b)] = {t: float((pred[t] == y_te[t]).mean()) for t in TASKS}
print('sliding 1-µs windows (centre µs):\n         ' + ' '.join(f'{0.5*b+0.5:5.1f}' for b in range(CFG['nb'] - 1)))
for t in TASKS: print(f'  {t:7s}' + ' '.join(f'{100*res["sliding"][str(b)][t]:5.1f}' for b in range(CFG['nb'] - 1)))
res['switching_test'] = dict(earliest_switch_ns=float(TF[ok].min()) if ok.any() else None, switches_mean=float(NSW.mean()),
                             frac_never=float((~ok).mean()))
res['readout_minutes'] = (time.time() - t1) / 60
print(f'colour errors in the two 9-mV pairs: {100*res["colour_err_in_pairs"]:.1f}% (final: 82.2) | '
      f'+ Purple triad {100*res["colour_err_in_pairs_plus_purple"]:.1f}% (final: 92.0)')
if not SMOKE: print('ALL WINDOWS PASS' if all(res['checks'].values()) else 'SOME WINDOWS DIFFER — send me the output')
json.dump(res, open(f'{OUT}/result_sigma0_60k.json', 'w'), indent=1); print(f'✓ saved {OUT}/result_sigma0_60k.json')
del R; gc.collect()
if not KEEP_FEATURES:
    for f in ('X_train.npy', 'X_test.npy', 'X_train.npy.progress', 'X_test.npy.progress'):
        if os.path.exists(f'{OUT}/{f}'): os.remove(f'{OUT}/{f}')
# Logistic readout on these features: use paper1/kaggle/logistic_sigma0_kaggle.py (current version).
