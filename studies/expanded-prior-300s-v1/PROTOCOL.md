# Five-minute CPU-limit sensitivity evaluation

Follow-up requested after examining the 120-second evaluation. This is a post-hoc resource-limit sensitivity, not a new untouched confirmatory test.

- Same 30 held-out families, 96 shared configurations, prepared train/validation splits, objective seed, XGBoost binary, seven policies, 10 optimizer seeds and budget 32 as `expanded-prior-v1`.
- Keep the historical prior byte-for-byte unchanged; no retraining or policy tuning.
- Raise the per-fit training CPU guard from 120 to 300 seconds. Keep 2,000 rounds and 100-round early-stopping patience. Preparation and model export remain outside the training guard. It is checked after each boosting round and may overshoot.
- Reuse all 2,812 completed fits verbatim. Restart all 68 censored fits from scratch, with the same inputs. Do not resume an early-stopping state from truncated models.
- Use 32 physical cores, one CPU thread per training job. CPU-time limits can allow slightly different round counts under different machine load; the guard is not a fixed-round budget.
- Verify that each retrained learning-curve prefix agrees with its original prefix, before replaying.
- Keep the original primary scoring: a still-censored fit is penalized, rather than credited with its best prefix. Compute the finite-pool optimum and normalization from completed 300-second results. Thus absolute regret across the two limits can change with the reference optimum/scale.
- Rerun all seven policies with identical seeds. Retain family-level bootstrap intervals and paired comparisons from the original analysis. Test losses remain unused.
- Publish/store only a delta of the 68 replaced evaluations and their curves, plus full new replay trajectories and reports. The original snapshot supplies unchanged records and splits.

Run with the original study environment and XGBoost binary:

```sh
python studies/expanded-prior-300s-v1/run.py
```

The local execution requires the original prepared data in `runs/expanded-prior-v1`. Public original data provenance and preparation scripts remain in the parent study. New and original frozen artifacts are never overwritten.
