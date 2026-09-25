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

## 2026-09-25 — Five-minute training-limit sensitivity started

User requested repeating the held-out evaluation with a300CPU-second per-fit limit and32cores. New immutable study: `studies/expanded-prior-300s-v1`, identity `596b1be90c686eade8f1e4da45e4ec0cb0bfb3a67c8d851327cfbde8ed3000bb`. Reuse2,812completed fits verbatim; restart68censored fits (67Dilbert,1mfeat-factors), retaining the prior, candidates, splits, seeds, patience, round limit, optimizer policies and scoring. Retrained curve prefixes must match before replay. The controller automatically runs all2,100optimizer paths and exports a compact replacement-record delta. Local launch/progress/logs: `runs/expanded-prior-300s-v1`.

This is an explicitly post-hoc resource sensitivity. Remaining timeouts still receive the original failure penalty; best-prefix scoring is not silently substituted. Changing completed pool membership can change the optimum and regret denominator, so before/after regret is not an absolute common-scale loss comparison.

## 2026-09-25 — Dilbert removed; five-minute default adopted

At the user's request, the current held-out comparison excludes Dilbert and selects the first eligible unused multiclass family in the original reserved order, without inspecting optimizer performance for selection. New study: `studies/expanded-prior-replacement-v1`, freeze `e19e0719c3c8554ce6348a2f7f2fa5950e4bd7d315c938b6a0276f1dadde1e2e`. Reuse 2,783 original completed fits and the successful 300-second mfeat-factors/eval_011 rerun; train only the replacement's 96 configurations on up to 32 cores. Prior, policy, candidate pool, seeds and failure scoring are unchanged.

The preceding 68-fit rerun finished in 730.6 seconds / 5.49 job CPU-hours: 10 complete, 58 still censored. Its replay was halted by an overly strict round-count audit: one 300-second Dilbert fit trained fewer rounds than at 120 seconds, while the common loss prefix matched. More CPU allowance does not guarantee more rounds under changed concurrency/resource contention. The superseded attempt is documented in its STATUS.md; it has no completed optimizer results.

For future work, `collection_defaults.json` sets 300 CPU-seconds per fit and `scripts/collect.py` applies and records that default inside each worker. Root README and AGENTS.md direct new studies to this entry point. Immutable historical scripts retain their original settings solely for archival reproduction.

## 2026-09-25 — Replacement evaluation completed

- Replaced Dilbert with tamilnadu-electricity (OpenML 40985): first eligible unused multiclass family in the frozen original order; 11,999 training rows, two features, 20 classes. Prior-training families remain disjoint. The new suite retains 30 families.
- Trained only 96 new fits in 15.0 seconds / 152.2 job CPU-seconds on up to 32 cores; reused 2,784 completed records, including the successful five-minute mfeat-factors update. All 2,880 current records are complete, with zero timeouts/failures.
- Replayed 2,100 optimizer trajectories in 10.9 seconds. Historical-prior GP early area 0.06189, no-history GP 0.07321, online GP-EI 0.07729, SMAC 0.08050, TPE 0.08046, random 0.07470. Prior early regret is 15.5% lower than no-history (95% family-bootstrap interval 4.6%–26.3% lower; Holm p=0.0489). SMAC has the lowest budget-32 point estimate, 0.01090.
- This is a post-hoc benchmark revision: confidence intervals and nominal p-values do not account for the adaptive dataset exclusion. The replacement has much lower feature dimensionality than Dilbert. Do not describe this as an untouched confirmation or general superiority.
- Saved the roughly 19 MB public data overlay (97 records: 96 new plus one updated), exact replacement split indices, frozen policies, reports and chart under `data/heldout-expanded-replacement-v1` and `studies/expanded-prior-replacement-v1`. Original snapshots remain unchanged.
- Verified all 2,880 records / 2,100 paths; all 1,960 paths on unchanged datasets are identical. A separate public-data replay reproduced 210/210 proposal paths. No extra objective training for these checks.
