# Interpretation of the revised held-out evaluation

Dilbert (OpenML 41163) was removed at the user’s request because of excessive runtime. It was replaced by tamilnadu-electricity (OpenML 40985), the first unused multiclass family in the original reserved order to pass the original admission rules. Selection did not use optimizer performance. The replacement has 11,999 training rows, two features and 20 classes; 20,000 rows were retained from 45,781 original rows. It does not preserve Dilbert’s high-dimensional workload characteristics.

The active suite has 30 families (13 binary, 12 multiclass, five regression) and 96 candidate configurations per family. All 2,880 evaluations now complete, with no CPU-censored or failed records. The same historical prior, candidate pool, splits for retained datasets, seven policies and optimizer seeds were reused.

## Results

| Method | Mean regret, evaluations 1–16 | Regret at evaluation 32 |
|---|---:|---:|
| GP + historical prior | 0.06189 | 0.01457 |
| GP without history | 0.07321 | 0.01628 |
| GP + prior, matched starts | 0.06798 | 0.01613 |
| GP-EI, online kernel | 0.07729 | 0.01438 |
| SMAC (pool) | 0.08050 | 0.01090 |
| Optuna TPE (pool) | 0.08046 | 0.01662 |
| Random | 0.07470 | 0.02148 |

Lower is better. Regret uses each completed finite pool’s minimum and max–min range; datasets receive equal weight after averaging ten optimizer seeds.

The historical-prior GP has the lowest early-area point estimate. Compared with the no-history GP, early regret is 15.5% lower (95% family-bootstrap interval: 4.6%–26.3% lower), with 22 wins among 30 families and Holm-adjusted paired sign-flip p=0.0489. It is 19.9% lower than the online-kernel GP-EI, approximately 23.1% lower than SMAC and TPE, and 17.1% lower than random search. All six early-area comparisons have nominal Holm-adjusted p<0.05 on this revised suite.

At evaluation 32, SMAC has the lowest point estimate. The early-area tests do not establish endpoint significance or that the prior GP is best at every budget. The GP variants use two initial proposals; the online-kernel GP uses five shared uniform starts; SMAC and TPE use ten. The matched-start GP control separates part of the prior-initialization effect.

## Scope and audit

**The dataset exclusion was decided after inspecting runtime and optimizer results.** These intervals and p-values are conditional on the revised suite, fixed prior and candidate pool; they do not account for that adaptive benchmark choice. This is evidence for the revised workload population, not a new untouched confirmation of general superiority. Dataset families are disjoint from prior training, but were inspected in earlier research. Baselines are finite-pool adapters, not unrestricted package HPO. No target test losses were evaluated.

Only 96 new fits were trained: 15.0 seconds elapsed and 152.2 job CPU-seconds on up to 32 cores. The other 2,784 records were reused, including the completed five-minute mfeat-factors rerun. Optimizer replay used 10.9 seconds elapsed and 292.0 CPU-seconds, without further objective training.

All 1,960 replay paths on the 28 unaffected pools match the original proposals and observed losses exactly. The mfeat-factors learning-curve prefix matches its original run exactly. A separate public-overlay replay of 210 paths across an unchanged dataset, mfeat-factors and the replacement reproduced every archived proposal. Curve minima, data and source checksums, splits, family separation, parameters and aggregate metrics passed verification.

## Future collection

The new default is 300 training CPU-seconds per fit, read from `collection_defaults.json` by `scripts/collect.py` and included in run identity. A custom study worker explicitly sets the same guard. It checks after a boosting round and may overshoot; data preparation and model export are excluded. The 2,000-round ceiling and 100-round early-stopping patience remain. Historical scripts and snapshots retain their original settings for exact archival checks.
