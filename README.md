# xgboost-hpo

Research data and reproducible experiments for few-shot hyperparameter optimization of XGBoost.

The aim is to learn useful search priors from historical evaluations, improve the first few proposals on a new dataset, and keep adapting as new observations arrive. This is an independent research repository, not an official XGBoost API.

## Current evaluation and collection defaults

The current comparison replaces Dilbert with the first eligible unused multiclass dataset from the reserved pool and uses a **300 CPU-second training limit per fit**. The other 29 families, prior and optimizer policies are retained. See the [current protocol](studies/expanded-prior-replacement-v1/PROTOCOL.md), [reproduction guide](studies/expanded-prior-replacement-v1/README.md), and [research log](RESEARCH_LOG.md). The replacement is a documented post-hoc benchmark revision.

Use `scripts/collect.py` for new collections. Its default is read from [collection_defaults.json](collection_defaults.json), currently five minutes; `--fit-cpu-seconds` allows an explicit override. The historical `scripts/reproduce.py` retains the original two-minute limit for reproducing archived runs.

## Expanded-space prior comparison

The original expanded-space study trained a historical mean/variance/kernel prior from the first database and evaluated it on separate dataset families and 96 fresh configurations, with a two-minute training limit. It compared the GP with no-history and online-kernel GPs, SMAC, Optuna TPE, and random search.

See the [study protocol and reproduction guide](studies/expanded-prior-v1/README.md) and [results](studies/expanded-prior-v1/reports/RESULTS.md). These are finite-pool optimizer comparisons, with validation objectives on held-out families; target test losses are not used.

## First database snapshot

**50 OpenML datasets × 40 shared random configurations = 2,000 attempted XGBoost fits.**

| Result | Count |
|---|---:|
| Successful evaluations | 1,999 |
| Failed evaluations, retained explicitly | 1 |
| Binary / multiclass / regression datasets | 22 / 18 / 10 |
| Successful fits reaching the 2,000-round ceiling | 185 |

The snapshot includes configuration values, raw validation losses, timing and model-size summaries, train/validation learning curves, validation predictions, original record metadata, and exact source-row split indices. The database occupies about **78 MB (75 MiB)**. Source tables are identified by **OpenML ID and version** and downloaded when reproducing; trained model binaries are omitted and can be regenerated.

Data are deliberately bounded: at most 20,000 retained rows and approximately four million feature cells per dataset. Median training size is 2,704 rows; 17 datasets were reduced. These results describe small and medium tabular workloads, not unrestricted large-data performance. Validation losses are log loss for classification and RMSE for regression; do not compare their raw magnitudes across datasets.

## Start here

Use Python 3.12 on Linux for collection reproduction.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify.py
python examples/read_database.py
python examples/ask_tell.py
```

Verification and database inspection need no downloads or model training. The ask/tell example demonstrates the NumPy GP interface with an explicitly illustrative prior; it does not train a historical prior.

```python
import pandas as pd

results = pd.read_parquet("data/collection-2026-09-25/evaluations.parquet")
valid = results[results.status == "complete"]
print(valid[["dataset", "cid", "validation_loss", "train_cpu_seconds"]].head())
```

## Collect objective evaluations

Start with one configuration on a small dataset:

```bash
python scripts/collect.py --dataset-id 1464 --config-id random_000 --workers 1
```

Reconstruct all source splits without training:

```bash
python scripts/collect.py --all --prepare-only
```

Recollect all 2,000 evaluations with up to 32 concurrent, single-thread fits and the current five-minute limit:

```bash
python scripts/collect.py --all --workers 32 --output runs/full-reproduction
```

Outputs are separate from the published snapshot. Existing evaluations are reused only within a matching environment/code identity. A failed or censored record is never silently replaced; choose another output directory for retries. Create `runs/full-reproduction/STOP` to request stopping at an iteration boundary, and remove it before resuming. Preparation downloads are bounded by the downloader's network behavior. The portable runner has no automatic 24-hour dispatch deadline, unlike the original controller.

**Numerical reproduction caveat:** the original run used a locally snapshotted **XGBoost 3.5.0-dev** library. Its hash and reported source revision are archived, but the binary's exact build ancestry was not independently verified. Installing a released XGBoost reproduces the design, not necessarily bitwise-identical losses. See [reproduction details](docs/reproduction.md).

## Repository map

- [Data schema and provenance](docs/data.md)
- [Reproduction instructions and limitations](docs/reproduction.md)
- [Study state and continuation instructions](docs/study.md)
- [Research log](RESEARCH_LOG.md)
- [Original collection protocol](docs/original-collection-protocol.md)
- [Failure diagnostic](docs/failure-diagnostic.md)
- `scripts/`: portable reconstruction, collection, configuration mapping, integrity checks.
- `examples/`: loading the database and NumPy-only GP ask/tell.
- `data/collection-2026-09-25/`: immutable first collection.

Earlier optimizer comparisons informed this collection but are not all migrated. The versioned studies contain the fitted expanded-space prior, optimizer comparisons and subsequent dataset/runtime revisions.

## License and attribution

Repository code is licensed under Apache-2.0; see [LICENSE](LICENSE). Source datasets retain their own terms. Dataset IDs, URLs, reported licenses, class mappings, and target names are in `datasets.json`. OpenML's generic “Public” label is not treated as a new license grant. This repository does not redistribute the original feature tables. The code license does not relicense third-party data or derived records; consult source terms when redistributing those.
