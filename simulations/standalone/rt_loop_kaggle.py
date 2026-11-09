# =============================================================================
# R–T hysteresis of the VO2 model (Zhang et al. model.py) — Paper 1, Appendix figure
# Kaggle cell (CPU is enough, < 1 min). Needs model.py in the inputs. No reservoir simulation:
# one VO2 element is driven through prescribed temperature paths and R(T) is recorded.
# Output: /kaggle/working/figures/rt_loop.json  → send this file.
# =============================================================================
import os, sys, json, importlib, numpy as np, torch

mp = next(os.path.join(r, 'model.py') for b in ('/kaggle/working', '/kaggle/input')
          for r, _, fs in os.walk(b) if 'model.py' in fs)
sys.path.insert(0, os.path.dirname(mp)); import model; importlib.reload(model)
assert model.VO2.reversal.__name__ == 'reversal'                  # original reversal (F7)
torch.set_default_device('cpu'); torch.set_grad_enabled(False)

def sweep(path, n=600):
    """Drive one VO2 element along the temperature corners in `path` (K); record R (kΩ)."""
    v = model.VO2(1); v.initialize(path[0] - 0.1)                   # same start as reset_vo2
    T_out, R_out = [], []
    for T0, T1 in zip(path[:-1], path[1:]):
        for T in np.linspace(T0, T1, n)[1:]:
            t = torch.tensor([T], dtype=torch.float32)
            v.reversal(t)
            T_out.append(float(T)); R_out.append(float(v.R(t)))
    return T_out, R_out

PATHS = {
    'major':      [305.0, 370.0, 305.0],          # full heating then full cooling
    'minor_338':  [305.0, 338.0, 315.0, 370.0],   # heating reversed inside the transition
}
out = {name: dict(path=p, **dict(zip(('T_K', 'R_kOhm'), sweep(p)))) for name, p in PATHS.items()}

v = model.VO2(1)
out['params'] = dict(Tc=float(v.Tc), width_w=float(v.w), beta=float(v.beta), R0=float(v.R0),
                     Ea=float(v.Ea), gamma=float(v.gamma), Rm_kOhm=float(v.Rm) / 1000,
                     T_clamp=[305, 370], T_base=325, source='model.py (Zhang et al.)')

# checks: insulating resistance at the operating temperature and metallic resistance
T, R = np.array(out['major']['T_K']), np.array(out['major']['R_kOhm'])
up = np.arange(len(T)) < len(T) // 2
i325 = np.argmin(np.abs(T[up] - 325)); R325 = R[up][i325]
print(f'R(325 K, heating, insulating) = {R325:.2f} kΩ   (PAPER1_STATE: 51.84 kΩ)')
print(f'R at 370 K (metallic)         = {R[up][-1]:.3f} kΩ   (Rm = {out["params"]["Rm_kOhm"]:.3f} kΩ)')
T_half_up = T[up][np.argmin(np.abs(np.log(R[up]) - 0.5 * (np.log(R[up][0]) + np.log(R[up][-1]))))]
T_half_dn = T[~up][np.argmin(np.abs(np.log(R[~up]) - 0.5 * (np.log(R[~up][0]) + np.log(R[~up][-1]))))]
print(f'mid-transition: heating {T_half_up:.1f} K, cooling {T_half_dn:.1f} K, '
      f'width {T_half_up - T_half_dn:.2f} K   (model w = {out["params"]["width_w"]:.2f} K, Tc = {out["params"]["Tc"]:.2f} K)')
out['checks'] = dict(R_325_heating_kOhm=float(R325), T_mid_heating_K=float(T_half_up),
                     T_mid_cooling_K=float(T_half_dn))

os.makedirs('/kaggle/working/figures', exist_ok=True)
json.dump(out, open('/kaggle/working/figures/rt_loop.json', 'w'))
print('✓ saved /kaggle/working/figures/rt_loop.json')
