# =============================================================================
# T_C0 — C0 × NOISE AT THE MAIN PROTOCOL (60k) (Paper 1, novelty 5)
# =============================================================================
# QUESTION: does the novelty-5 result (faster charging ↔ noise robustness), so far only in the reduced
# protocol (20k, one noise seed), hold with the main protocol (60k)?
# Conditions, in priority order (each saved separately; a crash loses at most one):
#   1) c0_0.5_sigma3e-4   key cell: 20k gave 89.61 / 77.99 / 92.20, no neuron without switching
#   2) c0_1.0_sigma3e-4   key cell: 20k gave 80.60 / 56.76 / 85.58, 97 % never switch
#   3) c0_0.5_sigma0      optional: noise-free reference for C0 × 0.5 at 60k
#                          (C0 × 1.0 at σ = 0, 60k is the final run: 91.38 / 81.48 / 93.43)
# PREDICTION if novelty 5 holds at 60k: (1) stays close to its noise-free level; (2) loses most of its
# colour accuracy and almost all switching; (3) ≈ the final run (same noise-free accuracy for all C0).
#
# Core (dataset, reservoir, F1/F2/F7, P5 ridge) copied unchanged from R1_final_sigma0_60k.py;
# noise seeding as in R1/R2 (noise_seed = 0, so both C0 values see the same noise draws).
# Inputs: model.py + paper1_data.json. Internet ON. GPU T4. Runtime estimate printed after the first batch
# of each condition. Features are written to resumable memmaps and deleted after each readout (disk).
# HOW TO RUN: SMOKE = True interactively (~5 min) → send me the output; then SMOKE = False + Save & Run All.
# Outputs: /kaggle/working/c0_60k/<condition>.json
# =============================================================================
SMOKE = True
RUN_SIGMA0_REFERENCE = True          # condition 3; set False to save ~1/3 of the time
CFG = dict(dt=10, t_max=10_000, T_base=325, R0=12.0, cth=0.15, couple=0.02, Vmin=10.5, Vmax=12.2,
           nb=20, n_train=60_000, n_val=10_000, n_test=10_000, split_seed=42, noise_seed=0, batch=500)
if SMOKE: CFG.update(n_train=600, n_val=200, n_test=300, batch=100)
RUN_SET = [('c0_0.5_sigma3e-4', 0.5, 3e-4), ('c0_1.0_sigma3e-4', 1.0, 3e-4)]
if RUN_SIGMA0_REFERENCE: RUN_SET.append(('c0_0.5_sigma0', 0.5, 0.0))
OUT = '/kaggle/working/c0_60k' + ('_smoke' if SMOKE else '')
import os; os.makedirs(OUT, exist_ok=True)

# ======================= SHARED CORE (from R1_final_sigma0_60k.py, unchanged) =======================
import sys, gc, json, time, importlib, numpy as np, torch
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

def simulate(c, V, seed):
    """features = max VO2 current per 500-ns bin (layout neuron*20 + bin); switch = R < 0.2·R(t=0)."""
    dt = CFG['dt']; NS = CFG['t_max'] // dt; SPB = NS // CFG['nb']; B = V.shape[0]
    with torch.device(DEV):
        reset_vo2(c)
        c.set_input(V=torch.as_tensor(V, dtype=torch.float32))
        y = torch.stack([torch.zeros(B, 784), torch.full((B, 784), float(CFG['T_base']))], dim=1)
        R_init = c.VO2.R(y[:, 1].reshape(-1)).reshape(B, 784)
        feat = torch.full((B, 784, CFG['nb']), -float('inf'))
        metal_prev = torch.zeros(B, 784, dtype=torch.bool)
        n_sw = torch.zeros(B, 784, dtype=torch.int32); t_first = torch.full((B, 784), -1.0)
        torch.manual_seed(seed)
        for i in range(NS):
            Rt = c.VO2.R(y[:, 1].reshape(-1)).reshape(B, 784)
            metal = Rt < 0.2 * R_init; onset = metal & ~metal_prev; metal_prev = metal
            t_first[onset & (t_first < 0)] = i * dt; n_sw += onset.int()
            dy = c.step(0, y)
            feat[:, :, i // SPB] = torch.maximum(feat[:, :, i // SPB], c.IR)
            y = y + dy * dt
        assert torch.isfinite(feat).all(), 'non-finite features'
        return feat.reshape(B, -1).cpu().numpy(), n_sw.cpu().numpy(), t_first.cpu().numpy()

def extract(c0, sigma, V, offset, tag, path):
    """features for all rows of V (padded last batch) into a resumable memmap at `path`."""
    n, B = len(V), CFG['batch']; nb = (n + B - 1) // B; prog = path + '.progress'
    X = np.lib.format.open_memmap(path, mode='r+' if os.path.exists(path) else 'w+', dtype=np.float32,
                                  shape=(n, 784 * CFG['nb']))
    start = int(open(prog).read()) if os.path.exists(prog) else 0
    NSW = np.zeros((n, 784), np.int16); TF = np.full((n, 784), -1.0, np.float32)
    if start > 0 and tag.endswith('test'):
        start = 0                                   # switching statistics need every test batch
    if start < nb:
        c = make_reservoir(c0, sigma, B); t0 = time.time()
        for k in range(start, nb):
            i0, i1 = k * B, min((k + 1) * B, n); Vb = V[i0:i1]
            if i1 - i0 < B: Vb = np.vstack([Vb, np.full((B - (i1 - i0), 784), CFG['Vmin'], np.float32)])
            f, s, t = simulate(c, Vb, CFG['noise_seed'] * 10**7 + offset + k)
            X[i0:i1] = f[:i1 - i0]; NSW[i0:i1] = s[:i1 - i0]; TF[i0:i1] = t[:i1 - i0]
            if k == start:
                print(f'    {tag}: first batch {time.time()-t0:.1f} s → ~{(time.time()-t0)*(nb-start)/60:.0f} min', flush=True)
            if k % 20 == 0 or k == nb - 1:
                X.flush(); open(prog, 'w').write(str(k + 1))
        del c; torch.cuda.empty_cache() if DEV == 'cuda' else None
    return X, NSW, TF

TASKS = {'digit': 10, 'color': 10, 'parity': 2}
COLS, K_ = {}, 0
for t, m in TASKS.items(): COLS[t] = slice(K_, K_ + m); K_ += m
ALPHAS = 10.0 ** np.arange(-6, 6.01, 0.5)
RDEV = DEV

def onehot(y):
    Y = -np.ones((len(y['digit']), K_)); r = np.arange(len(Y))
    Y[r, y['digit']] = 1; Y[r, 10 + y['color']] = 1; Y[r, 20 + y['parity']] = 1
    return Y

class Ridge:
    """P5: standardise with fit-split stats (sd < 1e-6 masked), α per task on validation, refit on fit+val."""
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
        sub = lambda G: G.clone() if full else G[it[:, None], it]
        mf = self.S['fit'][it] / nf; yf = self.SY['fit'] / nf
        Gc = sub(self.G['fit']); Gc.addr_(mf, mf, alpha=-nf)
        ZYc = self.ZY['fit'][it] - nf * torch.outer(mf, yf)
        Zv = torch.tensor(self.Zval[:, idx], dtype=f64, device=RDEV) - mf
        va = {t: [] for t in TASKS}
        if len(idx) <= 6000:
            lam, Q = torch.linalg.eigh(Gc); lam = lam.clamp(min=0); Pm = Q.T @ ZYc; A = Zv @ Q; del Q
            for a in ALPHAS:
                Sc = A @ (Pm / (lam + a)[:, None]) + yf
                for t in TASKS: va[t].append(float((Sc[:, COLS[t]].argmax(1).cpu().numpy() == self.yval[t]).mean()))
            del A, Pm, lam
        else:
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
        return dict(alpha=best, alpha_at_grid_edge=edge, val_acc_curve=va, n_masked=int(self.const.sum())), pred

def colour_stats(yc, pc):
    cm = np.zeros((10, 10), int); np.add.at(cm, (yc, pc), 1)
    ix = {nm: i for i, nm in enumerate(NAMES)}; off = cm.sum() - np.trace(cm)
    pairs = sum(cm[ix[a], ix[b]] + cm[ix[b], ix[a]] for a, b in [('Red', 'DkGreen'), ('Green', 'Orange')])
    return dict(confusion_color=cm.tolist(), colour_err_in_pairs=float(pairs / max(off, 1)),
                per_class_color={NAMES[i]: float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(10)})
# ===================== end of shared core =====================

# ---------------- data ----------------
nTR, nTE = CFG['n_train'], CFG['n_test']
V_tr, V_te = volts(G_all[:nTR], YC_all[:nTR]), volts(G_te_all[:nTE], YC_te_all[:nTE])
y_tr, y_te = labels(YD_all[:nTR], YC_all[:nTR]), labels(YD_te_all[:nTE], YC_te_all[:nTE])
ink_te = V_te > CFG['Vmin'] + 1e-6

GRID20 = J['c0_sigma_grid_20k']['runs']
FINAL60 = J['sigma0_60k_final']['windows']['full (0-19)']['test_acc']
fmt = lambda a: ' / '.join(f'{100*a[t]:.2f}' for t in TASKS)

# ---------------- main loop ----------------
for name, c0, sigma in RUN_SET:
    jpath = f'{OUT}/{name}.json'
    if os.path.exists(jpath): print(f'{name}: done — skip'); continue
    print(f'\n===== {name}: C0 × {c0}, σ = {sigma:g}, main protocol =====', flush=True)
    t0 = time.time()
    Xtr, _, _ = extract(c0, sigma, V_tr, 0, f'{name} train', f'{OUT}/{name}_Xtr.npy')
    Xte, NSW, TF = extract(c0, sigma, V_te, 10**6, f'{name} test', f'{OUT}/{name}_Xte.npy')
    t_sim = (time.time() - t0) / 60
    ok = TF >= 0
    sw = dict(switches_mean=float(NSW.mean()), frac_never=float((~ok).mean()),
              frac_ge3_switches=float((NSW >= 3).mean()),
              t_first_median_ink_ns=float(np.median(TF[ok & ink_te])) if (ok & ink_te).any() else None,
              t_first_median_bg_ns=float(np.median(TF[ok & ~ink_te])) if (ok & ~ink_te).any() else None,
              t_first_min_ns=float(TF[ok].min()) if ok.any() else None)
    print(f'    switching (test): mean {sw["switches_mean"]:.2f} | never {100*sw["frac_never"]:.1f} % | '
          f'≥3 switches {100*sw["frac_ge3_switches"]:.1f} % | simulation {t_sim:.0f} min', flush=True)
    t1 = time.time(); R = Ridge(Xtr, y_tr, Xte, CFG['n_val'])
    r, pred = R.window(np.arange(784 * CFG['nb'])); del R; gc.collect()
    acc = {t: float((pred[t] == y_te[t]).mean()) for t in TASKS}
    res = dict(name=name, cond=dict(c0=c0, sigma=sigma), smoke=SMOKE, tau_el_pred_ns=1416.32309693658 * c0,
               protocol=dict(**CFG, readout='P5 ridge, full window, α per task on val, refit on fit+val'),
               test_acc=acc, switching_test=sw, minutes=dict(sim=t_sim, readout=(time.time() - t1) / 60),
               reference_20k=GRID20[name]['test_acc'] if name in GRID20 else None,
               reference_20k_switching=GRID20[name]['switching_test'] if name in GRID20 else None,
               **r, **colour_stats(y_te['color'], pred['color']))
    print(f'    {name} (60k): {fmt(acc)}' + (f'  ⚠ α edge {r["alpha_at_grid_edge"]}' if r['alpha_at_grid_edge'] else ''))
    if name in GRID20: print(f'    same cell, reduced protocol (20k): {fmt(GRID20[name]["test_acc"])}')
    json.dump(res, open(jpath, 'w'), indent=1); print(f'✓ saved {jpath}', flush=True)
    del Xtr, Xte, NSW, TF; gc.collect()
    for f in (f'{name}_Xtr.npy', f'{name}_Xte.npy', f'{name}_Xtr.npy.progress', f'{name}_Xte.npy.progress'):
        if os.path.exists(f'{OUT}/{f}'): os.remove(f'{OUT}/{f}')

# ---------------- summary ----------------
print('\n========== SUMMARY (test accuracy %, digit / colour / parity; main protocol) ==========')
print(f'{"c0_1.0_sigma0 (final run)":28s} {fmt(FINAL60)}')
done = {}
for name, _, _ in RUN_SET:
    p = f'{OUT}/{name}.json'
    if os.path.exists(p):
        d = json.load(open(p)); done[name] = d
        print(f'{name:28s} {fmt(d["test_acc"])} | never switching {100*d["switching_test"]["frac_never"]:.1f} %')
if not SMOKE and 'c0_0.5_sigma3e-4' in done and 'c0_1.0_sigma3e-4' in done:
    a5, a10 = done['c0_0.5_sigma3e-4']['test_acc'], done['c0_1.0_sigma3e-4']['test_acc']
    base5 = done['c0_0.5_sigma0']['test_acc'] if 'c0_0.5_sigma0' in done else FINAL60
    holds = (base5['color'] - a5['color'] < 0.03) and (a5['color'] - a10['color'] > 0.10)
    print('VERDICT:', 'novelty 5 HOLDS at 60k (C0 × 0.5 keeps its colour accuracy at σ = 3e-4, C0 × 1.0 loses it)'
          if holds else 'NOT as predicted — send me the output')
