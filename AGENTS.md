# Research continuation

Read README.md, docs/study.md, docs/reproduction.md and RESEARCH_LOG.md before starting experiments.

- Treat data/collection-2026-09-25 as immutable. Create a versioned collection for new records.
- Never replace failed/censored records with successful retries without retaining the original and recording the retry policy.
- Preserve source/target dataset-family separation. Do not tune on a claimed held-out panel.
- Respect the user's latest CPU budget; otherwise default to one worker. Keep one thread per objective fit.
- Do not train models when only inspecting data. Use scripts/verify.py for integrity checking.
- Record protocols, dependency/build identities, seeds, resource caps, compute usage and conclusions.
- Avoid claims of exact numerical reproduction with a different XGBoost binary.
- Keep test losses out of hyperparameter selection.
- Append meaningful study changes to RESEARCH_LOG.md.
