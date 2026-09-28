# Expanded-space historical prior: held-out comparison

30 target families, 96 fresh candidate configurations, budget32,10optimizer seeds. Source:50families/1,999successful records at40distinct configurations.

| Method | Area 1–16 | Regret 4 | Regret 8 | Regret 16 | Regret 32 |
|---|---:|---:|---:|---:|---:|
| GP + historical prior | 0.06189 | 0.07849 | 0.04544 | 0.02874 | 0.01457 |
| GP without history | 0.07321 | 0.09259 | 0.05692 | 0.03201 | 0.01628 |
| GP + prior, matched starts | 0.06798 | 0.08150 | 0.04965 | 0.03161 | 0.01613 |
| GP-EI, online kernel | 0.07729 | 0.09826 | 0.06292 | 0.03536 | 0.01438 |
| SMAC (pool) | 0.08050 | 0.09826 | 0.06907 | 0.03574 | 0.01090 |
| Optuna TPE (pool) | 0.08046 | 0.09826 | 0.06907 | 0.03703 | 0.01662 |
| Random | 0.07470 | 0.09095 | 0.05819 | 0.03722 | 0.02148 |

Lower is better. Regret is relative to the completed finite-pool optimum, normalized by its max–min. See sensitivity.csv for median scaling and complete-pool-only results.

## Paired comparisons

Negative differences favor the historical GP. Intervals resample dataset families after averaging seeds.

| Comparator | Relative area difference (%) | 95% interval (%) | Holm p |
|---|---:|---|---:|
| gp_no_history | -15.5 | [-26.3, -4.6] | 0.0489 |
| gp_prior_matched2 | -9.0 | [-15.4, -2.6] | 0.0489 |
| gp_ei_online | -19.9 | [-31.8, -7.7] | 0.0186 |
| smac | -23.1 | [-35.3, -10.3] | 0.0102 |
| tpe | -23.1 | [-35.1, -10.4] | 0.0120 |
| random | -17.1 | [-29.8, -4.1] | 0.0489 |

## Compute and scope

Reused 2784 completed fits and trained 96 replacement-dataset fits in 0.3 minutes using up to32cores; 0.04 job CPU-hours. Status counts: {'complete': 2880, 'censored': 0, 'failed': 0}. Replayed 2100 optimizer trajectories with no additional objective training.

This is a held-out-family transfer experiment for this prior, not a wholly untouched research benchmark. Package baselines are finite-pool adapters; startup budgets differ as specified in PROTOCOL.md. All methods use the same legal pool. No target test losses were evaluated.
