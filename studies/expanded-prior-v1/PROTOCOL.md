# Expanded-space prior transfer: frozen evaluation v1

2026-09-25. Authorized: train a prior on the new 50-family collection and compare the GP with common HPO choices on held-out datasets. Maximum 16 physical cores, one thread per objective fit. No cloud resources.

## Source fit

Use all 1,999 successful evaluations from the immutable 50-family/40-configuration source snapshot. Retain the one failed record as missing, never fill it with its diagnostic retry. Normal midrank scores are computed separately within each family's observed configurations (39 for the incomplete family). Equal total weight per family. Mean and log residual variance are fitted by two 200-tree XGBoost regressors (depth3, eta.05, hist128, lambda1), preserving the earlier recipe and total fitting weight200. Scale floor .05. Fit a Matérn5/2 ARD covariance with18 lengthscales, amplitude and noise by three fixed source-only likelihood starts. Missing records contribute only co-observed pairs. Source fitting and all code are frozen before target outcomes.

Encoding reuses the earlier effective18-dimensional mapping: log learning rate; depth active/value; leaves active/logvalue; growth policy; log min_child_weight; subsample; effective tree/level/node sampling strengths; active/logvalue pairs for L2,L1,gamma; log2 max_bin. No inactive random generator coordinates. This is an expanded-space adaptation of the earlier GP, not the previously evaluated five-parameter fitted model. Forty source locations may be insufficient to estimate18-dimensional structure reliably; do not tune it on target results.

## Targets and collection

From the59 previously reserved families, choose one dataset per family (lowest OpenML ID), deterministic metadata-hash order. Seek30 admitted families:12binary,11multiclass,7regression, filling exhausted quotas from other types. Same admission and retained-size policy as the source collection:100–500k raw rows,≤5000features,≤30million rawcells,≤50classes; cap20kretainedrows/four million cells; missing-target removal, native categoricals, train-only category vocabulary, numeric infinities to missing, duplicate-feature-group-disjoint approximately60/20/20 split. Only admission failures can cause replacements, never objective quality. Archive all exclusions. If fewer30 qualify, report that count and do not silently introduce another dataset pool.

These families are held out from this prior fit. They have been inspected in earlier, different benchmark experiments; this is a fresh expanded-space generalization experiment, not a claim of wholly untouched research-level confirmation. Test-split labels are never scored. Evaluation here means the held-out datasets' validation objectives, not held-out test losses.

Train96 new shared IID configurations, seed20260927, using exactly the source parameter mapping/ranges. None may equal a source configuration. Each target/configuration pair is trained once, with the same recorded3.5.0-dev binary as the historical source,2000rounds,patience100,120CPU-secondguard. Thus the maximum is2880 new objective fits. The old development binary's ancestry limitation remains. Curves/metadata/models are retained locally; portable artifacts omit models/source tables.

For comparison, treat any failed or CPU-censored evaluation as an unsuccessful query: count its budget and return a finite penalty above that dataset's worst completed value (worst + max(range,1%abs(worst),1e-12)). The oracle computes this penalty only; no pool bounds or unqueried losses are supplied to policies. This uses benchmark-side scale information for the failure signal and must be disclosed. Report counts and a complete-pool-only sensitivity if any failures/censoring occur. Round-ceiling hits without a CPU guard remain valid finite-fidelity outcomes.

## Optimizers, fixed before outcomes

All see the same legal96-candidate pool. Budget32,10random seeds/dataset. Store every proposal and queried value. Native optimizer initial designs/acquisition maximizers are replaced by finite-pool procedures; do not label this an unrestricted end-to-end package benchmark.

1. **GP + historical prior:** NumPy Q2 online policy: two prior Thompson proposals then normal-rank expected improvement of latent candidate–incumbent differences. No portfolio or epsilon random.
2. **GP without history:** same Q2 policy, zero mean, unit-scale fixed isotropic Matérn5/2, lengthscale1, noise.1.
3. **GP + prior, matched starts:** learned GP with the shared first two uniform proposals. Compare with method1 to isolate the effect of prior-guided starts; generic GP's first two Thompson proposals need not match these.
4. **GP-EI, online kernel:** scikit-learn GP, five shared uniform starts, standardized current losses,18ARD Matérn5/2 plus amplitude/noise, one MLE start per fit,100L-BFGS iterations, latent-variance EI relative to best observed standardized loss.
5. **SMAC (pool):** SMAC2.4.1 HPO-facade RF, runhistory transformation, EI and default random-design injection. Ten shared uniform starts. Score all remaining legal configurations; nonbinding RF max-leaf limit set to observation count for efficiency.
6. **Optuna TPE (pool):** Optuna5.0 independent TPE density-ratio scoring across the same18 effective coordinates; ten shared uniform starts; native Parzen estimators, gamma and prior defaults. This replaces candidate generation, not density fitting. Conditional numeric features use active indicators and canonical inactive zeros.
7. **Random:** uniform without replacement.

Two/five/ten startup budgets are explicit policy choices, not additional free evaluations. Shared uniform prefixes are identical across the baselines where used. Report early-budget differences with that distinction. No baseline or prior hyperparameter sweep. Prior-fit costs and online optimizer costs are recorded separately from objective evaluation costs.

Official package references: https://automl.github.io/SMAC3/latest/ and https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html .

## Analysis

Primary: dataset-normalized simple validation regret averaged over evaluations1–16. Normalize by completed96-pool max−min; constant pools contribute zero for successful queries and are explicitly counted. Regret is relative to this finite pool, not a global optimum. A robustness table uses distance to the completed-pool median when its denominator is positive, since extrema can be sensitive to poor configurations. Secondary: regrets2,4,8,16,32 and average rank vs budget. Always report classification/regression strata.

Average10optimizer seeds within each dataset before paired10,000-replicate dataset-family bootstrap intervals. Report pointwise95%bands for curves; paired differences against learned GP, and Holm-adjusted paired sign-flip tests for the primary metric across six comparisons. Mark differences under5%relative early-area as practically small, independently of statistical significance. This threshold is a reporting aid, not optimizer selection. Also report cost of actually queried fits and optimizer CPU time. Do not bootstrap individual queries as independent datasets.

No test-set peeking, source/target family overlap, filling missing losses from unrelated runs, or changing methods after seeing outcomes. Code corrections must be logged with their effect on the freeze and whether any results were rerun.
