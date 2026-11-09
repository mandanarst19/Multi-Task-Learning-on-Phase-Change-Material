import json, sys
U='/mnt/user-data/uploads/'
D=json.load(open(U+'23c2093d-paper1_data.json'))
sub=json.load(open(U+'7d3cf8d2-result_subthreshold_60k__1_.json'))
prec=json.load(open(U+'7ff936b8-result_sub_precision_60k__3_.json'))
c0={n:json.load(open(U+f)) for n,f in [('c0_0.5_sigma0','210af4b8-c0_0.5_sigma0__2_.json'),
                                       ('c0_0.5_sigma3e-4','0544973a-c0_0.5_sigma3e-4__1_.json'),
                                       ('c0_1.0_sigma3e-4','93d662fa-c0_1.0_sigma3e-4__1_.json')]}
assert not sub['smoke'] and not prec['smoke'] and all(not v['smoke'] for v in c0.values())
for k in ('subthreshold_mechanism_60k','sub_precision_60k','c0_noise_60k'): assert k not in D
D['subthreshold_mechanism_60k']={
  'source':'file: result_subthreshold_60k.json (Kaggle, T_subthreshold_mechanism_60k.py, 2026-11-07), full run, P5 60k, sigma 0, C0 x1.0',
  'definition':'features = max per 500-ns bin over 0-2 us (bins 0-3); (a) full model current I, (b) full model device voltage V1, (c) FROZEN-T: dT forced to 0 (T = T_base, R constant, I proportional to input voltage). float32 simulation.',
  'reproduction':'(a) reproduces sigma0_60k_final windows bins 0-3 and cum 0-1us: PASS',
  'verdict_printed':sub['verdict'],
  'verdict_note':'NOT as predicted in float32 because of a rounding artefact in the linear (frozen) model; resolved by sub_precision_60k',
  'heating_before_2us':sub['heating_before_2us'],
  'results':{s:{w:{'test_acc':r['test_acc'],'alpha':r['alpha'],'alpha_at_grid_edge':r['alpha_at_grid_edge'],'n_masked':r['n_masked']}
              for w,r in v.items()} for s,v in sub['results'].items()},
  'use_in_paper':'heating medians 6.1 K (ink) / 5.3 K (background) in Results 3.2'}
P={'source':'file: result_sub_precision_60k.json (Kaggle, T_sub_precision_60k.py, 2026-11-07), full run, P5 60k, sigma 0, C0 x1.0',
  'definition':'FULL and FROZEN-T over 0-2 us and 0-1 us, simulated in float32 and float64 (features kept in float64 for fp64); ridge in float64; test curve = refit on fit+val, test accuracy at every alpha',
  'alpha_grid':prec['alpha_grid'],'verdict':prec['verdict'],'results':{}}
for run,v in prec['results'].items():
  P['results'][run]={w:{k:r[k] for k in ('test_acc_selected','alpha_selected','alpha_at_grid_edge','n_masked','val_curve','test_curve')} for w,r in v.items()}
R=prec['results']
P['key_numbers']={'full_fp64_0_2us':R['full fp64']['0-2us']['test_acc_selected'],
  'frozen_fp64_0_2us':R['frozen fp64']['0-2us']['test_acc_selected'],
  'frozen_fp32_0_2us_colour_at_alpha_1e-6':R['frozen fp32']['0-2us']['test_curve']['color'][0],
  'max_abs_diff_full_fp32_vs_fp64_selected_pp':round(100*max(abs(R['full fp32'][w]['test_acc_selected'][t]-R['full fp64'][w]['test_acc_selected'][t]) for w in ('0-2us','0-1us') for t in ('digit','color','parity')),3)}
P['interpretation']='Frozen (linear) model in float64 = raw-input level (22.3 %) at every alpha; float32 reaches 50.6 % only at alpha 1e-6 (rounding artefact). Full model fp32 = fp64 within 0.1 pp: the 52.5 % sub-threshold colour accuracy is physical (Joule heating + R(T)). Supports Results 3.2 and Methods 2.6 (frozen-temperature control).'
D['sub_precision_60k']=P
C={'source':'files: c0_60k/*.json (Kaggle, T_c0_noise_60k.py, 2026-11-08), full runs, P5 60k (50k fit / 10k val), full window only, noise_seed 0',
  'purpose':'confirm novelty 5 (c0_sigma_grid_20k) in the main protocol',
  'reference_c0_1.0_sigma0':'sigma0_60k_final (91.38 / 81.48 / 93.43)','runs':{}}
for n,v in c0.items():
  C['runs'][n]={k:v[k] for k in ('cond','tau_el_pred_ns','test_acc','switching_test','alpha','alpha_at_grid_edge','val_acc_curve','n_masked','confusion_color','colour_err_in_pairs','per_class_color','reference_20k')}
C['verdict']='novelty 5 holds at 60k: C0x0.5 loses <=0.72 pp at sigma 3e-4 (0 % never-switching); C0x1.0 loses 8.4 / 20.9 / 6.3 pp (97 % never switch). No alpha at grid edge; selected alpha 1e3-1e5 (no small-alpha artefact).'
C['observation_not_claim']='C0x1.0 at sigma 3e-4: colour errors no longer concentrated in the two pairs (35 %); Purple 26 %, Gray 64 %.'
D['c0_noise_60k']=C
D['_readme']['updated']='2026-11-08: added subthreshold_mechanism_60k, sub_precision_60k, c0_noise_60k (all provenance level: file)'
json.dump(D,open('paper1_data.json','w'),indent=1,ensure_ascii=False)
E=json.load(open('paper1_data.json'))
print(list(E)[-4:]); print(E['sub_precision_60k']['key_numbers'])
print({n:E['c0_noise_60k']['runs'][n]['test_acc'] for n in E['c0_noise_60k']['runs']})
