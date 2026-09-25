# Research log

## 2026-09-25 — Initial public repository

Created `xgboost-hpo` to preserve the latest expanded-space collection in a portable form and enable continuation by another researcher or agent.

- Exported 50 OpenML dataset manifests, exact source-row train/validation/test splits, 40 shared IID configurations, and 2,000 evaluation records.
- Kept 1,999 successful loss/prediction archives and the original MiceProtein failure. The diagnostic rerun remains separate.
- Stored scalar results in Zstandard-compressed Parquet and full records in gzip JSONL; grouped existing compressed NPZ curves by dataset. Omitted approximately 1.7 GB of trained models and original source tables.
- Added portable OpenML reconstruction and bounded collection scripts, an integrity verifier, and NumPy GP ask/tell examples.
- Preserved original collection protocol/scripts and recorded build provenance. Documented that released XGBoost reruns need not match the unverified development binary exactly.
- Public scope is the fresh 50-dataset collection and GP API example. Earlier full optimizer studies are not yet migrated.

Validation results are recorded in `docs/validation.json` and `docs/reconstruction-check.json`. Future experiments should append dated entries with protocol, inputs, compute budget, output location, findings and remaining uncertainty.

## 2026-09-25 — Expanded-space prior / held-out evaluation started

User authorized learning a prior from the newly collected results and comparing the GP against common alternatives on held-out datasets. New study: `studies/expanded-prior-v1/`; immutable source database unchanged.

- Historical prior fitted to 1,999 successful observations from 50 families at 40 unique configurations; one source failure remains missing. Two XGBoost mean/scale fits plus three fixed ARD kernel optimization starts, approximately 0.28 CPU seconds. No target losses read.
- Reused the existing 18-dimensional effective-parameter encoding. Seven kernel coordinates reached optimization bounds; this is a limitation to evaluate, not a reason to tune on target outcomes.
- Frozen policy/collection identity: `bab6051d8e3d514637fe5c284ef74a2c4c45215007c9121deb98379176c68127`. Target30families from the reserved59,96new IID configurations per family, budget32,10optimizer seeds. Maximum2,880objective fits,16physical cores,one thread per fit.
- Frozen comparisons: learned-prior GP, no-history GP, learned GP with matched uniform starts, current-task kernel-fit GP-EI, SMAC pool adapter, independent Optuna TPE pool adapter, random search.
- Prespecified primary mean normalized regret at budgets1–16, endpoint32, rank/regret curves, family-bootstrap uncertainty, task-type strata and cost/failure accounting. Source/target families and exact configurations disjoint.
- Kernel gradient and baseline synthetic checks passed. Prior projection API exactly reproduces the frozen candidate prior. Reporting/export/API helper code was added during collection before reviewing target performance; those helpers do not affect policy decisions and are recorded in the final artifact manifest.

## 2026-09-25 — Expanded-space evaluation completed

-30held-out families (13binary,12multiclass,5regression),96fresh configurations each;2,880objective fits in1,407.6seconds/5.16jobCPU-hours on16cores.2,812complete,68CPU-censored,0failed;289complete round-cap hits.67censored fits wereDilbert and1mfeat-factors.
-2,100trajectories/67,200queries replayed in19.0seconds,279.8CPU seconds;0additionalobjectivefits.
-Primaryearlyarea: historicalGP0.08453,no-historyGP0.08035,onlineGP0.08988,SMAC0.09413,TPE0.09429,random0.08561. No robust overall superiority; prior wins22/30families but suffers a large Dilbert penalty.
-Predeclared28complete-pool sensitivity: prior15.5%lower earlyregret than no-historyGP,95%improvementCI[4.5%,26.1%]. Conditional sensitivity, not a replacement for the primary result. Prior-guided initial proposals improve the same model relative to two uniform starts; primaryHolmp=.033.
-Meanrank/regret charts, objective-type strata, pairedbootstrap intervals, cost tables, normalization sensitivity, prior diagnostics and censoring diagnosis saved under studies/expanded-prior-v1/reports.
-Published compact target data under data/heldout-expanded-v1 (about123MB), no original tables/models. Added source-ID/split reconstruction via --snapshot and no-training replay from public records. New guard bookkeeping retained censored prefixes and avoided failed assertions when the native best_iteration was stale.
-All source and policy freezes retained; no policy or prior was changed after target outcomes. Results motivate a separately versioned timeout/failure-handling experiment, not silent tuning of this comparison.

Publication checks: the compact database, all2,100replay paths, source/policy freeze and final artifact manifest passed verification. A separate public-table replay of140trajectories (one complete pool and Dilbert's censored pool) matched every archived proposal, without any new objective training. The generated figure was visually inspected. Forty-nine censored fits exhibited the stale native best-iteration condition; the collector preserved the correct curve-minimum prefix and censored status.
