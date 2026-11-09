# =============================================================================
# T_SUB — SUB-THRESHOLD MECHANISM TEST (Paper 1, novelty 3) — main protocol (60k), σ = 0
# =============================================================================
# QUESTION: is the colour gain before the first switch (cumulative 0–2 µs: 52.54 % vs 22.32 % for the
# raw input) caused by heating of the device changing its resistance R(T)?
#
# Three feature sets over 0–2 µs only (bins 0–3, max per 500-ns bin), same P5 ridge readout, 60k:
#   (a) FULL model, device current I     → must reproduce 85.54 / 52.54 / 89.04 (PASS check)
#   (b) FULL model, device voltage V1    → informative (V1 also feels R(T) through eq. 2)
#   (c) FROZEN-T model, current I        → temperature held at T_base (dT = 0), so R is constant and
#       I is exactly proportional to the input voltage.
#       PREDICTION: (c) falls to the raw-input level 85.00 / 22.32 / 88.99.
# If (c) falls to the raw-input level while (a) stays at 52.5 %, the sub-threshold colour gain is
# SHOWN to come from heating-induced changes of R (Results 3.2 can then drop "consistent with").
# Also reported: the 0–1 µs window (bins 0–1), and the peak temperature rise before 2 µs
# (ink vs background) as a physical number for the Discussion.
#
# Inputs: model.py + paper1_data.json. Internet ON. GPU T4.
# Only 200 time steps per image → fast (~10–20 min in total; estimate printed after the first batch).
# HOW TO RUN: SMOKE = True interactively (~2 min) → send me the output; then SMOKE = False + Save & Run All.
# Output: /kaggle/working/subthreshold/result_subthreshold_60k.json
# =============================================================================
SMOKE = True
CFG = dict(dt=10, t_max=10_000, T_base=325, R0=12.0, cth=0.15, couple=0.02, Vmin=10.5, Vmax=12.2,
           nb=20, n_train=60_000, n_val=10_000, n_test=10_000, split_seed=42, noise_seed=0, batch=500)
if SMOKE: CFG.update(n_train=600, n_val=200, n_test=300, batch=100)
NB_SUB = 4                                   # bins 0–3 = 0–2 µs (before the first switch at 2.03 µs)
OUT = '/kaggle/working/subthreshold' + ('_smoke' if SMOKE else '')
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
        lam, Q = torch.linalg.eigh(Gc); lam = lam.clamp(min=0); Pm = Q.T @ ZYc; A = Zv @ Q; del Q
        for a in ALPHAS:
            Sc = A @ (Pm / (lam + a)[:, None]) + yf
            for t in TASKS: va[t].append(float((Sc[:, COLS[t]].argmax(1).cpu().numpy() == self.yval[t]).mean()))
        del A, Pm, lam, Gc, ZYc, Zv
        best = {t: float(ALPHAS[int(np.argmax(va[t]))]) for t in TASKS}
        edge = [t for t in TASKS if best[t] in (ALPHAS[0], ALPHAS[-1])]
        nt = nf + self.n['val']; m_ = (self.S['fit'][it] + self.S['val'][it]) / nt; y_ = (self.SY['fit'] + self.SY['val']) / nt
        Gc = sub(self.G['fit']); Gc += sub(self.G['val']); Gc.addr_(m_, m_, alpha=-nt)
        ZYc = self.ZY['fit'][it] + self.ZY['val'][it] - nt * torch.outer(m_, y_)
        Zt = torch.tensor(self.Zte[:, idx], dtype=f64, device=RDEV) - m_; pred = {}
        for t in TASKS:
            W = self._solve(Gc, best[t], ZYc[:, COLS[t]]); pred[t] = (Zt @ W + y_[COLS[t]]).argmax(1).cpu().numpy(); del W
        del Gc, ZYc, Zt; torch.cuda.empty_cache() if DEV == 'cuda' else None
        return dict(alpha=best, alpha_at_grid_edge=edge, n_masked=int(self.const[idx].sum())), pred
# ===================== end of shared core =====================

# ---------------- sub-threshold simulation: 0–2 µs only ----------------
SPB = (CFG['t_max'] // CFG['dt']) // CFG['nb']          # 50 steps per 500-ns bin
NS_SUB = NB_SUB * SPB                                     # 200 steps = 2 µs

def simulate_sub(c, V, freeze_T):
    """Features over 0–2 µs, layout neuron*NB_SUB + bin:
       I = max device current per bin, V1 = max device voltage per bin.
       freeze_T=True: dT forced to 0 → T = T_base, R constant, I ∝ input voltage."""
    B = V.shape[0]
    with torch.device(DEV):
        reset_vo2(c)                                                                # F1
        c.set_input(V=torch.as_tensor(V, dtype=torch.float32))
        y = torch.stack([torch.zeros(B, 784), torch.full((B, 784), float(CFG['T_base']))], dim=1)
        R_init = c.VO2.R(y[:, 1].reshape(-1)).reshape(B, 784)
        fI = torch.full((B, 784, NB_SUB), -float('inf')); fV = torch.full((B, 784, NB_SUB), -float('inf'))
        metal_prev = torch.zeros(B, 784, dtype=torch.bool); n_sw = torch.zeros(B, 784, dtype=torch.int32)
        Tmax = torch.full((B, 784), float(CFG['T_base']))
        for i in range(NS_SUB):
            Rt = c.VO2.R(y[:, 1].reshape(-1)).reshape(B, 784)
            metal = Rt < 0.2 * R_init; n_sw += (metal & ~metal_prev).int(); metal_prev = metal
            dy = c.step(0, y)
            if freeze_T: dy[:, 1] = 0.0
            b = i // SPB
            fI[:, :, b] = torch.maximum(fI[:, :, b], c.IR)
            fV[:, :, b] = torch.maximum(fV[:, :, b], y[:, 0])
            y = y + dy * CFG['dt']
            Tmax = torch.maximum(Tmax, y[:, 1])
        assert torch.isfinite(fI).all() and torch.isfinite(fV).all(), 'non-finite features'
        return (fI.reshape(B, -1).cpu().numpy(), fV.reshape(B, -1).cpu().numpy(),
                n_sw.cpu().numpy(), Tmax.cpu().numpy())

def extract_sub(V, freeze_T, tag):
    n, B = len(V), CFG['batch']; d = 784 * NB_SUB
    XI = np.zeros((n, d), np.float32); XV = np.zeros((n, d), np.float32)
    NSW = np.zeros((n, 784), np.int16); TM = np.zeros((n, 784), np.float32)
    c = make_reservoir(1.0, 0.0, B); t0 = time.time()
    for k, i0 in enumerate(range(0, n, B)):
        i1 = min(i0 + B, n); Vb = V[i0:i1]
        if i1 - i0 < B: Vb = np.vstack([Vb, np.full((B - (i1 - i0), 784), CFG['Vmin'], np.float32)])
        fI, fV, s, tm = simulate_sub(c, Vb, freeze_T)
        XI[i0:i1], XV[i0:i1], NSW[i0:i1], TM[i0:i1] = fI[:i1 - i0], fV[:i1 - i0], s[:i1 - i0], tm[:i1 - i0]
        if k == 0:
            print(f'    {tag}: first batch {time.time()-t0:.1f} s → ~{(time.time()-t0)*n/B/60:.1f} min', flush=True)
    del c; torch.cuda.empty_cache() if DEV == 'cuda' else None
    return XI, XV, NSW, TM

def idx_bins(b0, b1): return (np.arange(784)[:, None] * NB_SUB + np.arange(b0, b1)[None, :]).ravel()

# ---------------- data ----------------
nTR, nTE = CFG['n_train'], CFG['n_test']
V_tr, V_te = volts(G_all[:nTR], YC_all[:nTR]), volts(G_te_all[:nTE], YC_te_all[:nTE])
y_tr, y_te = labels(YD_all[:nTR], YC_all[:nTR]), labels(YD_te_all[:nTE], YC_te_all[:nTE])
ink_te = V_te > CFG['Vmin'] + 1e-6

# ---------------- simulations ----------------
t0 = time.time()
XI_tr, XV_tr, _, _ = extract_sub(V_tr, False, 'FULL train')
XI_te, XV_te, NSW_full, TM_full = extract_sub(V_te, False, 'FULL test')
XF_tr, _, _, _ = extract_sub(V_tr, True, 'FROZEN-T train')
XF_te, _, NSW_frz, TM_frz = extract_sub(V_te, True, 'FROZEN-T test')
t_sim = (time.time() - t0) / 60
print(f'simulation {t_sim:.1f} min | switches before 2 µs: full {int(NSW_full.sum())}, frozen {int(NSW_frz.sum())} '
      f'(expected 0 and 0; first switch in the final run: 2030 ns)', flush=True)
dT_full = TM_full - CFG['T_base']
heat = dict(ink_median_K=float(np.median(dT_full[ink_te])), ink_p90_K=float(np.percentile(dT_full[ink_te], 90)),
            background_median_K=float(np.median(dT_full[~ink_te])),
            frozen_max_K=float((TM_frz - CFG['T_base']).max()))
print(f'peak temperature rise before 2 µs (full model, test set): ink median {heat["ink_median_K"]:.2f} K '
      f'(90th pct {heat["ink_p90_K"]:.2f} K), background median {heat["background_median_K"]:.2f} K | '
      f'frozen model max {heat["frozen_max_K"]:.2e} K (must be 0)', flush=True)

# ---------------- readouts ----------------
REF = J['sigma0_60k_final']['windows']
RAW = J['baselines_60k']['A_raw_input_linear']
WINS = {'0-2us (bins 0-3)': (0, 4, 'bins 0-3'), '0-1us (bins 0-1)': (0, 2, 'cum 0-1us')}
SETS = {'(a) full, current I': (XI_tr, XI_te), '(b) full, voltage V1': (XV_tr, XV_te),
        '(c) frozen-T, current I': (XF_tr, XF_te)}
res = dict(smoke=SMOKE, protocol=dict(**CFG, NB_SUB=NB_SUB, readout='P5 ridge (eigh), α per task on val, refit on fit+val'),
           simulation_minutes=t_sim, heating_before_2us=heat,
           references=dict(full_model_windows={w: REF[k]['test_acc'] for w, (_, _, k) in WINS.items()},
                           raw_input_linear={t: RAW[t] for t in TASKS}),
           results={})
t1 = time.time()
for sname, (Xa, Xb) in SETS.items():
    R = Ridge(Xa, y_tr, Xb, CFG['n_val']); res['results'][sname] = {}
    for wname, (b0, b1, refkey) in WINS.items():
        r, pred = R.window(idx_bins(b0, b1)); acc = {t: float((pred[t] == y_te[t]).mean()) for t in TASKS}
        r['test_acc'] = acc; res['results'][sname][wname] = r
        line = f'  {sname:26s} {wname:17s}: ' + ' | '.join(f'{t} {100*acc[t]:.2f}' for t in TASKS)
        if sname.startswith('(a)') and not SMOKE:
            ok = all(abs(acc[t] - REF[refkey]['test_acc'][t]) <= 0.006 for t in TASKS)
            r['reproduces_final_run'] = ok; line += f'  → REPRODUCTION {"PASS" if ok else "DIFFERENT — send me the output"}'
        print(line + (f'  ⚠ α edge {r["alpha_at_grid_edge"]}' if r['alpha_at_grid_edge'] else ''), flush=True)
    del R; gc.collect(); torch.cuda.empty_cache() if DEV == 'cuda' else None
res['readout_minutes'] = (time.time() - t1) / 60

# ---------------- verdict ----------------
print('\nreference — raw input (linear readout, 60k): ' + ' / '.join(f'{100*RAW[t]:.2f}' for t in TASKS))
print('reference — final run, 0–2 µs window:          ' + ' / '.join(f'{100*REF["bins 0-3"]["test_acc"][t]:.2f}' for t in TASKS))
if not SMOKE:
    c_frz = res['results']['(c) frozen-T, current I']['0-2us (bins 0-3)']['test_acc']['color']
    c_full = res['results']['(a) full, current I']['0-2us (bins 0-3)']['test_acc']['color']
    if abs(c_frz - RAW['color']) <= 0.03 and c_full - c_frz > 0.20:
        verdict = 'CONFIRMED: without heating the colour falls to the raw-input level → the gain comes from R(T)'
    else:
        verdict = 'NOT as predicted — send me the output'
    res['verdict'] = verdict; print('VERDICT:', verdict)
json.dump(res, open(f'{OUT}/result_subthreshold_60k.json', 'w'), indent=1)
print(f'✓ saved {OUT}/result_subthreshold_60k.json')
