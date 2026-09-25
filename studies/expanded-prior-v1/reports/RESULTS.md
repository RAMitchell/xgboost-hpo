# Expanded-space historical prior: held-out comparison

**Finding:** no established overall advantage over the no-history GP; a favorable early-budget effect on the 28 complete pools is offset by poor handling of time-capped workloads. See [interpretation](INTERPRETATION.md).

30 target families, 96 fresh candidate configurations, budget32,10optimizer seeds. Source:50families/1,999successful records at40distinct configurations.

| Method | Area 1–16 | Regret 4 | Regret 8 | Regret 16 | Regret 32 |
|---|---:|---:|---:|---:|---:|
| GP + historical prior | 0.08453 | 0.11237 | 0.07399 | 0.03467 | 0.01667 |
| GP without history | 0.08035 | 0.09769 | 0.05942 | 0.03325 | 0.01699 |
| GP + prior, matched starts | 0.09592 | 0.11604 | 0.08422 | 0.03709 | 0.01869 |
| GP-EI, online kernel | 0.08988 | 0.12212 | 0.06742 | 0.03767 | 0.01511 |
| SMAC (pool) | 0.09413 | 0.12212 | 0.07930 | 0.03708 | 0.01128 |
| Optuna TPE (pool) | 0.09429 | 0.12212 | 0.07930 | 0.03898 | 0.01667 |
| Random | 0.08561 | 0.10781 | 0.06757 | 0.03899 | 0.02209 |

Lower is better. Regret is relative to the completed finite-pool optimum, normalized by its max–min. See sensitivity.csv for median scaling and complete-pool-only results.

## Paired comparisons

Negative differences favor the historical GP. Intervals resample dataset families after averaging seeds.

| Comparator | Relative area difference (%) | 95% interval (%) | Holm p |
|---|---:|---|---:|
| gp_no_history | 5.2 | [-23.6, 47.7] | 1.0000 |
| gp_prior_matched2 | -11.9 | [-16.6, -4.8] | 0.0330 |
| gp_ei_online | -5.9 | [-29.5, 20.7] | 1.0000 |
| smac | -10.2 | [-32.8, 14.3] | 1.0000 |
| tpe | -10.4 | [-32.6, 13.8] | 1.0000 |
| random | -1.3 | [-26.6, 28.4] | 1.0000 |

## Compute and scope

Collected 2880 objective fits in 23.5 minutes using up to16cores; 5.16 job CPU-hours. Status counts: {'complete': 2812, 'censored': 68, 'failed': 0}. Replayed 2100 optimizer trajectories with no additional objective training.

This is a held-out-family transfer experiment for this prior, not a wholly untouched research benchmark. Package baselines are finite-pool adapters; startup budgets differ as specified in PROTOCOL.md. All methods use the same legal pool. No target test losses were evaluated.
