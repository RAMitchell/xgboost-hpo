# Replace Dilbert and use a five-minute training limit

The user requested removing Dilbert after its high timeout rate persisted at 300 CPU-seconds. This is a post-hoc revision of the benchmark population, not an untouched confirmation. Keep the original evaluations and the attempted 300-second Dilbert rerun archived; exclude Dilbert from this new comparison.

- Retain the other 29 families, all 96 candidate configurations, historical prior files, seven optimizer policies, 10 optimizer seeds and 32-query budget.
- Select the first unused multiclass family in the original reserved candidate order that passes the original metadata/preparation rules. The candidate order is frozen in `replacement_candidates.json`; do not inspect optimizer performance to select a replacement. If none is eligible, report the 29-family evaluation explicitly.
- Continue to enforce no family overlap with the 50 historical training datasets. These evaluation families were inspected in earlier research, so they are held out from this prior rather than wholly untouched research data.
- Reuse 2,783 completed original fits and the newly completed 300-second `mfeat-factors/eval_011` fit. Validate that its original learning-curve prefix agrees exactly. Train 96 configurations only for the replacement dataset.
- Use up to 32 physical cores, one CPU thread per fit, a 300-second training CPU guard, 2,000 boosting rounds, and 100-round early-stopping patience. Preparation and export are outside the training guard; the guard checks after each round and can overshoot.
- Keep the original failure penalty for any still-censored or failed fit. No best-prefix substitution, objective change, prior refit or policy tuning.
- Replay all policies and retain the original normalized-regret/rank analysis, dataset-family bootstrap and paired tests. Recompute the pool reference from completed fits. Do not evaluate test losses.
- Store a compact delta containing the replacement dataset's 96 records and the updated mfeat-factors record. Reuse the parent snapshot for unchanged records and splits.

This study freezes a 300-second limit. `collection_defaults.json` and the current `scripts/collect.py` entry point make that limit the default for future collection. Original scripts and snapshots retain their historical settings for exact archival verification.
