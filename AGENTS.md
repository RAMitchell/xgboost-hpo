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

## Current collection settings

- The default training CPU limit is **300 seconds per fit**, recorded in `collection_defaults.json`. Use `scripts/collect.py` for new collections; it applies the limit in every worker and includes it in run identity. The latest user-authorized concurrency for this study is 32 physical cores, with one thread per fit.
- `scripts/collector.py` and `scripts/reproduce.py` retain their original 120-second values solely for archived experiment verification. Do not use those historical defaults for new experiments. Custom study workers must explicitly apply the current limit.
- Keep frozen prior models and policy files unchanged, version follow-ups, and reuse unaffected fits.
- Dilbert (OpenML 41163) is excluded from the current held-out evaluation at the user's request because of excessive runtime. Its historical records remain archived. Do not silently reintroduce it.
- The current study is `studies/expanded-prior-replacement-v1`. Held-out means disjoint from prior training families; many evaluation families were inspected in earlier research. Report this limitation and the post-hoc dataset replacement.
