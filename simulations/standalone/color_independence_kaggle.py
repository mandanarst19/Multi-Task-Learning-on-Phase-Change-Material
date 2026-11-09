# Colored MNIST: digit–color independence (labels only, no simulation). CPU, Internet ON, ~1 min.
# Paper 1 → paper1_data.json: colour_label_independence. Run 2026-11-07; colour assignment verified
# (test colour counts equal the row sums of sigma0_60k_final.confusion_color: PASS).
import os, json, numpy as np
from scipy.stats import chi2_contingency
from torchvision import datasets

np.random.seed(42)                                   # same color assignment as every paper run
yc_tr = np.random.randint(0, 10, 60000)
yc_te = np.random.randint(0, 10, 10000)
yd_tr = datasets.MNIST('/kaggle/working/data', train=True,  download=True).targets.numpy()
yd_te = datasets.MNIST('/kaggle/working/data', train=False, download=True).targets.numpy()

# check: test color counts must equal the row sums of the paper's final color confusion matrix
ref = None
for r, _, fs in os.walk('/kaggle/input'):
    if 'paper1_data.json' in fs:
        J = json.load(open(os.path.join(r, 'paper1_data.json')))
        ref = np.array(J['sigma0_60k_final']['confusion_color']).sum(1)
if ref is None:   # fallback: the same reference, hard-coded (test images per colour in the paper)
    ref = np.array([1021, 1030, 988, 1071, 969, 1008, 967, 956, 1008, 982])
assert (np.bincount(yc_te, minlength=10) == ref).all(), 'color assignment differs from the paper'
print('color assignment check: PASS')

def test(a, b, ka, kb, name):
    tab = np.zeros((ka, kb), int); np.add.at(tab, (a, b), 1)
    chi2, p, dof, exp = chi2_contingency(tab, correction=False)
    n = int(tab.sum()); V = float(np.sqrt(chi2 / (n * (min(ka, kb) - 1))))
    print(f'{name:30s} n={n:6d}  chi2({dof}) = {chi2:8.2f}  p = {p:.3f}  V = {V:.4f}  min E = {exp.min():.1f}')
    return dict(n=n, chi2=float(chi2), dof=int(dof), p=float(p), cramers_v=V,
                min_expected=float(exp.min()), table=tab.tolist())

R = {}
for split, yd, yc in (('train_60k', yd_tr, yc_tr),
                      ('train_20k', yd_tr[:20000], yc_tr[:20000]),
                      ('test_10k',  yd_te, yc_te)):
    R[f'digit_x_color_{split}']  = test(yd, yc, 10, 10, f'digit x color, {split}')
    R[f'parity_x_color_{split}'] = test(yd % 2, yc, 2, 10, f'parity x color, {split}')
    t = np.array(R[f'parity_x_color_{split}']['table'])
    R[f'parity_from_color_best_acc_{split}'] = float(t.max(0).sum() / t.sum())   # best parity guess from colour alone
    print(f'    best parity accuracy from colour alone ({split}): {100*R[f"parity_from_color_best_acc_{split}"]:.2f}%')
R['method'] = ('Pearson chi-square test of independence (scipy chi2_contingency, no continuity correction); '
               'Cramer V = sqrt(chi2 / (n (k - 1))), k = min(rows, cols)')
R['color_generation'] = 'np.random.seed(42); randint(0,10,60000) train, then randint(0,10,10000) test'
json.dump(R, open('/kaggle/working/color_independence.json', 'w'), indent=1)
print('saved /kaggle/working/color_independence.json')
