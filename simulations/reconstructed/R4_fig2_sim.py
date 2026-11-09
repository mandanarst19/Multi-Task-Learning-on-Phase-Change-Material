# =============================================================================
# R4 — fig2_sim.npz: σ = 0 SWITCHING SIMULATION (RECONSTRUCTION)
# Used by make_fig1.py (idxA, VA, IA, ydA, ycA, meta) and make_fig3.py (VA, IA, VB, onsetB, tfirstB, meta).
# (Review 2026-11-02: idxA added — make_fig1.py reads z['idxA']; the old file stores idxA directly.)
# =============================================================================
# RECONSTRUCTED, NOT THE ORIGINAL. Group A = the 10 Fig-1 images (full current traces, float16);
# group B = the first 1000 test images (input voltages, first-switch time per neuron, switch onsets per step).
# The 10 Fig-1 images: if the OLD fig2_sim.npz is in the inputs (from paper1_figures_final.zip), their
# indices are recovered by matching VA to the test voltages, and the new file is compared with the old one.
# Otherwise set IDX_A by hand (it must contain test image 78: digit 9, Yellow). ~10 min on a T4.
# =============================================================================
SMOKE = False                 # already small
IDX_A = None                  # e.g. [ ..., 78, ... ] — only needed when the old fig2_sim.npz is not in the inputs
N_B = 1000
CFG = dict(dt=10, t_max=10_000, T_base=325, R0=12.0, cth=0.15, couple=0.02, Vmin=10.5, Vmax=12.2,
           nb=20, split_seed=42, noise_seed=0, batch=500)
OUT = '/kaggle/working/fig2_sim'; import os; os.makedirs(OUT, exist_ok=True)
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

V_te = volts(G_te_all, YC_te_all)
OLD = None
try:
    p_old = find('fig2_sim.npz'); OLD = np.load(p_old, allow_pickle=False); print('old fig2_sim.npz found:', p_old)
except FileNotFoundError:
    print('old fig2_sim.npz not in the inputs')
if OLD is not None:
    IDX_A = [int(i) for i in OLD['idxA']] if 'idxA' in OLD.files else \
            [int(np.argmin(np.abs(V_te - v[None, :].astype(np.float32)).max(1))) for v in OLD['VA']]
    err = max(float(np.abs(V_te[i] - v).max()) for i, v in zip(IDX_A, OLD['VA']))
    print(f'Fig-1 images recovered from the old file: {IDX_A} (max |ΔV| = {err:.2e} V)')
    assert err < 1e-3, 'old VA rows do not match the test voltages'
assert IDX_A is not None and 78 in IDX_A, 'set IDX_A (must contain test image 78)'

# ---- group A: full traces ----
cA = make_reservoir(1.0, 0.0, len(IDX_A))
oA = simulate(cA, V_te[IDX_A], 0, trace=True); del cA
# ---- group B: first N_B test images ----
B = CFG['batch']; onset = None; TFB = []
cB = make_reservoir(1.0, 0.0, B)
for k, i0 in enumerate(range(0, N_B, B)):
    i1 = min(i0 + B, N_B); Vb = V_te[i0:i1]
    if i1 - i0 < B: Vb = np.vstack([Vb, np.full((B - (i1 - i0), 784), CFG['Vmin'], np.float32)])
    o = simulate(cB, Vb, k)
    TFB.append(o['t_sw'][:i1 - i0, :, 0])
    if i1 - i0 < B: raise ValueError('use N_B divisible by batch (onset counts would include padding)')
    onset = o['onset_t'] if onset is None else onset + o['onset_t']
del cB
TFB = np.concatenate(TFB)
meta = dict(palette=P.tolist(), V=[CFG['Vmin'], CFG['Vmax']], dt=CFG['dt'], t_max=CFG['t_max'], c0=1.0, sigma=0.0,
            couple=CFG['couple'], idx_A=IDX_A, idx_B=[0, N_B], switch_rule='R < 0.2 R(t=0)', reconstructed=True)
np.savez_compressed(f'{OUT}/fig2_sim.npz', idxA=np.array(IDX_A), VA=V_te[IDX_A], IA=oA['I'], ydA=YD_te_all[IDX_A], ycA=YC_te_all[IDX_A],
                    VB=V_te[:N_B], onsetB=onset.astype(np.int32), tfirstB=TFB.astype(np.float32), meta=json.dumps(meta))
ok = TFB >= 0
print(f'earliest switch in group B: {TFB[ok].min():.0f} ns (60k test set: 2030 ns) | never switch {100*(~ok).mean():.1f}%')
if OLD is not None:
    for k in ('VB', 'tfirstB', 'onsetB'):
        a, b = OLD[k].astype(float), np.load(f'{OUT}/fig2_sim.npz')[k].astype(float)
        print(f'  compare {k}: shapes old {a.shape} new {b.shape}' + (f' | max |Δ| {np.abs(a - b).max():.3g}' if a.shape == b.shape else ''))
    a, b = OLD['IA'].astype(float), oA['I'].astype(float)
    print(f'  compare IA: shapes old {a.shape} new {b.shape}' + (f' | max |Δ| {np.abs(a - b).max():.3g} mA' if a.shape == b.shape else ''))
print(f'✓ saved {OUT}/fig2_sim.npz')
