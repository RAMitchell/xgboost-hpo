# Study state and handoff

## Latest completed study

The expanded-space prior has now been trained and evaluated: see [the study](../studies/expanded-prior-v1/README.md) and [interpretation](../studies/expanded-prior-v1/reports/INTERPRETATION.md). The 30-family comparison shows useful early transfer on the 28 complete candidate pools but no robust overall advantage when time-capped fits are penalized. All 2,880 objective evaluations and 2,100 replay paths are preserved. Prior fitting, projection, replay, and reporting scripts are now portable. No experiment remains running.

The material below records the context and plan preceding that completed study; proposed steps 1–4 have now been performed in expanded-prior-v1. The next priority is explicitly testing failure/cost handling while preserving this result.

## Research objective

Build a practical few-shot XGBoost hyperparameter optimizer using compact historical information and modest dependencies. Optimize validation regret against evaluation budget, while tracking actual compute cost and avoiding a heavily tuned optimizer whose own configuration becomes another HPO problem.

Historical evaluations should guide the earliest proposals while allowing current-task evidence to dominate later. Priors should transfer across classification/regression loss scales. Preserve dataset-family separation between prior training and evaluation.

## Where the research currently stands

Earlier work compared random search, established Bayesian optimizers, XGBoost surrogates, portfolio starts, explicit exploration, quadratic Bayesian linear regression, and GP acquisition variants. The current GP prototype uses task-relative normal rank scores, a caller-supplied historical mean/covariance, two prior Thompson proposals, then expected improvement using the uncertainty of candidate-minus-incumbent differences. `examples/quantile_gp_numpy.py` extracts its online behavior. Historical mean/scale learning and kernel fitting are **not implemented by that example**, and no default learned prior is bundled in this initial repository.

Historical-data ablations on the earlier TabRepo pool suggested that about 20–40 configurations per source dataset captured much of the observed prior benefit. Shared IID sampling was selected for simplicity: more complex designs had no robust advantage. A one-fold source-prior ablation had a 0.33% point estimate of early-regret degradation relative to three folds, with a 95% interval from -1.46% to +2.16% on the primary 54-source comparison. This motivated collecting one source split, while retaining more rigorous evaluation on independent target tasks.

Increasing historical source datasets from 54 to 120 on the earlier five-parameter pool produced only a 0.21% point improvement in early regret, with a 95% interval spanning -3.56% to +4.11%. That does not establish universal saturation: the candidate pool, parameter ranges and inspected target panel limit the conclusion.

The new collection expands the parameter space and records training curves and realized ensemble structure. It has **not yet established an effective historical prior in that expanded space**. The fresh 50-family snapshot is the first portable artifact here. Earlier full comparison scripts, priors and trace databases remain in the original local study and have not all been migrated; the numbers above are context, not results reproducible from this repository alone.

## Important limitations

- Forty configurations are shared across source datasets; they are not dense coverage of the expanded search space.
- Small/medium retained dataset sizes, one source split, 2,000-round ceiling, and a CPU guard define the fidelity.
- Fifty source families include only ten regression datasets after recorded admission failures.
- Source metadata, including a generic license label, can be incomplete.
- The original development binary's compilation ancestry is not independently established.
- One original fit failed. The successful diagnostic rerun must not silently replace it.
- Family labels reduce known overlap but cannot prove all dataset aliases have been eliminated.

## Next work, not yet executed

1. Train expanded-space priors using only source families; specify the encoding of conditional growth/column-sampling parameters.
2. Obtain target evaluations on separate dataset families and candidate configurations; freeze this evaluation protocol before selecting among priors.
3. Compare the current GP with and without historical information, random search, and a common Bayesian baseline using equal evaluation budgets and controlled initialization.
4. Plot regret and mean rank against budget, with uncertainty resampled at dataset-family level; report early-budget area, endpoint regret, failure/censoring and wall time.
5. Investigate whether more diverse datasets, more configurations, or more fidelity produces the best gain per collection CPU hour.
6. Before any confirmatory collection, freeze a clean released/build-pinned XGBoost environment.

Do not run additional training merely by reading this handoff. New experimental changes should produce a versioned directory, explicit protocol, and log entry.
