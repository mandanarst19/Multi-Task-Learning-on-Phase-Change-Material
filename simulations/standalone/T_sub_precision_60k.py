# =============================================================================
# T_SUB_PREC — NUMERICAL-PRECISION CHECK of the sub-threshold colour gain (Paper 1, novelty 3)
# main protocol (60k), σ = 0, window 0–2 µs (bins 0–3) and 0–1 µs (bins 0–1)
# =============================================================================
# WHY: in T_SUB (2026-11-07) the FROZEN-T model — temperature held at T_base, so R is constant and every
# feature is exactly (input voltage) × (constant per bin) — still reached 50.5 % colour (raw input: 22.3 %),
# with the colour α at the LOWER grid edge (1e-6). In exact arithmetic this is impossible.
# HYPOTHESIS: float32 rounding in the Euler integration leaves a tiny, input-dependent 'fingerprint'
# that a weakly regularised ridge exploits.
# TEST: FULL and FROZEN-T, each simulated in float32 AND float64 (features kept in float64 for fp64).
# Reported per run and window: test accuracy at the validation-selected α, the whole validation curve,
# and the test accuracy at every α of the grid (so fixed α = 1e-2, 1, 100 can be compared).
# PREDICTIONS if the hypothesis is right:
#   FROZEN-T fp64 → colour ≈ 22 % (raw-input level) at every α
#   FROZEN-T fp32 → colour high only at small α, falling to ≈ 22 % at large α
#   FULL fp64     → the true (physical) sub-threshold colour gain
# Inputs: model.py + paper1_data.json. Internet ON. GPU T4. ~10–15 min.
# HOW TO RUN: SMOKE = True (~2 min) → check it runs; then SMOKE = False + Save & Run All. Send the output.
# Output: /kaggle/working/subprec/result_sub_precision_60k.json
# =============================================================================
SMOKE = True
CFG = dict(dt=10, t_max=10_000, T_base=325, R0=12.0, cth=0.15, couple=0.02, Vmin=10.5, Vmax=12.2,
           nb=20, n_train=60_000, n_val=10_000, n_test=10_000, split_seed=42, noise_seed=0, batch=500)
if SMOKE: CFG.update(n_train=600, n_val=200, n_test=300, batch=100)
NB_SUB = 4
OUT = '/kaggle/working/subprec' + ('_smoke' if SMOKE else '')
import os; os.makedirs(OUT, exist_ok=True)

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

# ---------------- dataset (verified, unchanged) ----------------
J = json.load(open(find('paper1_data.json')))
NAMES = J['sigma0_60k_final']['color_order']
assert NAMES == ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'Orange', 'Purple', 'DkGreen', 'Gray']
P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
              [1, .5, 0], [.5, 0, 1], [0, .5, 0], [.5, .5, .5]], np.float32)
L = P @ np.array([0.299, 0.587, 0.114], np.float32)
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

# ---------------- reservoir in a chosen precision ----------------
def reset_vo2(c):                                                                   # F1
    v = c.VO2; T0 = torch.full((v.N,), CFG['T_base'] - 0.1, device=DEV)
    v.delta = torch.ones(v.N, device=DEV); v.reversed = torch.zeros(v.N, device=DEV)
    v.gr = v.g_major(T0); v.Tr = T0.clone(); v.Tpr = v.Tpr_func(); v.T_last = T0.clone()

def make_reservoir(B):
    with torch.device(DEV):
        c = model.Circuit2D(B, 28, 28, 0.0, CFG['R0'], 0.0, CFG['cth'], CFG['couple'], 1.0, CFG['T_base'])
    c.R0C0 = c.R0 * c.C0                                                           # F2 (C0 × 1.0)
    return c

SPB = (CFG['t_max'] // CFG['dt']) // CFG['nb']; NS_SUB = NB_SUB * SPB

def simulate_sub(c, V, freeze_T, dtype):
    B = V.shape[0]
    with torch.device(DEV):
        reset_vo2(c)
        c.set_input(V=torch.as_tensor(V, dtype=dtype))
        y = torch.stack([torch.zeros(B, 784, dtype=dtype), torch.full((B, 784), float(CFG['T_base']), dtype=dtype)], dim=1)
        fI = torch.full((B, 784, NB_SUB), -float('inf'), dtype=dtype)
        for i in range(NS_SUB):
            dy = c.step(0, y)
            if freeze_T: dy[:, 1] = 0.0
            b = i // SPB
            fI[:, :, b] = torch.maximum(fI[:, :, b], c.IR)
            y = y + dy * CFG['dt']
        assert c.IR.dtype == dtype, f'internal dtype {c.IR.dtype} != {dtype}'
        assert torch.isfinite(fI).all()
        return fI.reshape(B, -1).cpu().numpy()

def extract_sub(V, freeze_T, dtype, tag):
    old = torch.get_default_dtype(); torch.set_default_dtype(dtype)        # model.py builds tensors with the default dtype
    try:
        n, B = len(V), CFG['batch']; X = np.zeros((n, 784 * NB_SUB), np.float64 if dtype == torch.float64 else np.float32)
        c = make_reservoir(B); t0 = time.time()
        for k, i0 in enumerate(range(0, n, B)):
            i1 = min(i0 + B, n); Vb = V[i0:i1]
            if i1 - i0 < B: Vb = np.vstack([Vb, np.full((B - (i1 - i0), 784), CFG['Vmin'], np.float32)])
            X[i0:i1] = simulate_sub(c, Vb, freeze_T, dtype)[:i1 - i0]
            if k == 0: print(f'    {tag}: first batch {time.time()-t0:.1f} s → ~{(time.time()-t0)*n/B/60:.1f} min', flush=True)
        del c; torch.cuda.empty_cache() if DEV == 'cuda' else None
    finally:
        torch.set_default_dtype(old)
    return X

# ---------------- P5 ridge, all α evaluated on validation AND test (eigh; d ≤ 3136) ----------------
TASKS = {'digit': 10, 'color': 10, 'parity': 2}
COLS, K_ = {}, 0
for t, m in TASKS.items(): COLS[t] = slice(K_, K_ + m); K_ += m
ALPHAS = 10.0 ** np.arange(-6, 6.01, 0.5)
f64 = torch.float64

def onehot(y):
    Y = -np.ones((len(y['digit']), K_)); r = np.arange(len(Y))
    Y[r, y['digit']] = 1; Y[r, 10 + y['color']] = 1; Y[r, 20 + y['parity']] = 1
    return Y

def ridge_all_alpha(Xtr, ytr, Xte, yte, idx):
    """P5: standardise (fit stats, sd<1e-6 masked); α per task on validation; refit on fit+val.
    Everything in float64. Returns selected-α test acc, validation curve, test curve over all α."""
    Xtr, Xte = np.asarray(Xtr[:, idx], np.float64), np.asarray(Xte[:, idx], np.float64)
    n = len(Xtr); is_val = np.zeros(n, bool)
    is_val[np.random.RandomState(CFG['split_seed']).permutation(n)[:CFG['n_val']]] = True
    fit = ~is_val
    mu = Xtr[fit].mean(0); sd = Xtr[fit].std(0); const = sd < 1e-6; sd[const] = 1.0
    T_ = lambda X: torch.tensor(np.where(const, 0.0, (X - mu) / sd), dtype=f64, device=DEV)
    Y = torch.tensor(onehot(ytr), dtype=f64, device=DEV)
    def solve_curve(Za, Ya, Zb, yb):
        m = Za.mean(0); ym = Ya.mean(0); Zc = Za - m
        lam, Q = torch.linalg.eigh(Zc.T @ Zc); lam = lam.clamp(min=0)
        Pm = Q.T @ (Zc.T @ (Ya - ym)); A = (Zb - m) @ Q
        curve = {t: [] for t in TASKS}
        for a in ALPHAS:
            S = A @ (Pm / (lam + a)[:, None]) + ym
            for t in TASKS: curve[t].append(float((S[:, COLS[t]].argmax(1).cpu().numpy() == yb[t]).mean()))
        return curve
    Zf, Zv = T_(Xtr[fit]), T_(Xtr[is_val])
    va = solve_curve(Zf, Y[torch.tensor(fit, device=DEV)], Zv, {t: ytr[t][is_val] for t in TASKS})
    te = solve_curve(T_(Xtr), Y, T_(Xte), yte)                    # refit on fit+val, test at every α
    best = {t: int(np.argmax(va[t])) for t in TASKS}
    sel = {t: te[t][best[t]] for t in TASKS}
    del Zf, Zv; torch.cuda.empty_cache() if DEV == 'cuda' else None
    return dict(test_acc_selected=sel, alpha_selected={t: float(ALPHAS[best[t]]) for t in TASKS},
                alpha_at_grid_edge=[t for t in TASKS if best[t] in (0, len(ALPHAS) - 1)],
                val_curve=va, test_curve=te, n_masked=int(const.sum()))

# ---------------- run ----------------
nTR, nTE = CFG['n_train'], CFG['n_test']
V_tr, V_te = volts(G_all[:nTR], YC_all[:nTR]), volts(G_te_all[:nTE], YC_te_all[:nTE])
y_tr, y_te = labels(YD_all[:nTR], YC_all[:nTR]), labels(YD_te_all[:nTE], YC_te_all[:nTE])
WINS = {'0-2us': (0, 4), '0-1us': (0, 2)}
idx_bins = lambda b0, b1: (np.arange(784)[:, None] * NB_SUB + np.arange(b0, b1)[None, :]).ravel()
RUNS = [('full fp32', False, torch.float32), ('frozen fp32', True, torch.float32),
        ('full fp64', False, torch.float64), ('frozen fp64', True, torch.float64)]
SHOW = {'1e-2': int(np.argmin(abs(ALPHAS - 1e-2))), '1': int(np.argmin(abs(ALPHAS - 1))), '100': int(np.argmin(abs(ALPHAS - 100)))}
res = dict(smoke=SMOKE, protocol=dict(**CFG, NB_SUB=NB_SUB), alpha_grid=ALPHAS.tolist(), results={},
           references=dict(raw_input=J['baselines_60k']['A_raw_input_linear'],
                           final_run_0_2us=J['sigma0_60k_final']['windows']['bins 0-3']['test_acc']))
t0 = time.time()
for name, frz, dt_ in RUNS:
    Xa = extract_sub(V_tr, frz, dt_, f'{name} train'); Xb = extract_sub(V_te, frz, dt_, f'{name} test')
    res['results'][name] = {}
    for wn, (b0, b1) in WINS.items():
        r = ridge_all_alpha(Xa, y_tr, Xb, y_te, idx_bins(b0, b1)); res['results'][name][wn] = r
        c = r['test_curve']['color']
        print(f'  {name:12s} {wn}: selected ' + ' / '.join(f'{100*r["test_acc_selected"][t]:.2f}' for t in TASKS)
              + f' (α colour {r["alpha_selected"]["color"]:.0e})'
              + ' | colour test at α=' + ', '.join(f'{k}: {100*c[i]:.1f}' for k, i in SHOW.items())
              + (f'  ⚠ edge {r["alpha_at_grid_edge"]}' if r['alpha_at_grid_edge'] else ''), flush=True)
    del Xa, Xb; gc.collect()
res['minutes'] = (time.time() - t0) / 60
print('\nreference — raw input (linear): 85.00 / 22.32 / 88.99 | final run 0–2 µs: 85.54 / 52.54 / 89.04')
if not SMOKE:
    g = lambda n, w='0-2us': res['results'][n][w]['test_acc_selected']['color']
    print('full fp32 reproduces final run 0–2 µs:', 'PASS' if abs(g('full fp32') - 0.5254) <= 0.006 else 'DIFFERENT')
    if g('frozen fp64') < 0.26:
        v = ('HYPOTHESIS CONFIRMED: the linear (frozen) model falls to the raw-input level in float64 → '
             'the fp32 gain was a rounding artefact. Physical sub-threshold colour accuracy = full fp64 = '
             f'{100*g("full fp64"):.2f} %.')
    else:
        v = 'HYPOTHESIS NOT CONFIRMED: the frozen model stays above the raw input in float64 — send me the output'
    res['verdict'] = v; print('VERDICT:', v)
json.dump(res, open(f'{OUT}/result_sub_precision_60k.json', 'w'), indent=1)
print(f'✓ saved {OUT}/result_sub_precision_60k.json')
