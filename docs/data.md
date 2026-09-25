# Database format

Snapshot: `data/collection-2026-09-25/`. All paths below are relative to that directory.

| File | Meaning |
|---|---|
| `evaluations.parquet` | One row per attempted fit; Zstandard compressed, typed columns. |
| `records.jsonl.gz` | Full original record content, including per-tree structure arrays, errors and hashes; machine-specific study path prefixes replaced by `{ORIGINAL_STUDY}`. |
| `datasets.json` | OpenML IDs, versions, URLs, source MD5s, reported licenses, family labels, features, classes, retained sizes and original data identities. |
| `configurations.json` | Exact 40 parameter dictionaries and underlying 13-dimensional IID unit vectors. |
| `splits/<uid>.npz` | `train`, `validation`, `test`: zero-based positions in the original OpenML source table, before dropping missing targets or subsampling. Order is significant. |
| `curves/<uid>.zip` | One member `<record_id>.npz` per successful fit. Existing compressed NPZ bytes are preserved exactly. |
| `dataset_sizes.csv` | Human-readable source and retained size audit. |
| `candidates.json`, `admissions.json` | Original candidate ordering and admissions/exclusions, including transient download failures. |
| `reserved_families.json` | Families excluded from the historical-source pool. |
| `completion.json`, `execution.json` | Original counts and compute accounting; state remains `incomplete` because one fit failed. |
| `provenance/` | Original environment, freeze, hardware, diagnostic results and archived scripts. |
| `manifest.json` | SHA-256 checksums for packaged snapshot files. |

## Evaluation columns

`uid` identifies a dataset/split, `cid` a shared configuration, and `record_id` their combination. `status` is `complete` or `failed` in this snapshot; future collectors can also emit `censored`. A failed fit has missing loss/curve fields, not an invented penalty loss. Exactly one record, `openml_40966_r0__random_014`, failed.

`param_*` columns contain the sampled XGBoost parameters. The full effective objective, seed and thread settings are also in the full JSONL record. `validation_loss` is the minimum observed validation loss; `best_iteration` is zero-based and `selected_rounds = best_iteration + 1`. `trained_rounds` includes early-stopping patience. `validation_loss_independent` was recomputed from selected-model predictions. `validation_baseline` is the training-mean/class-frequency predictor's validation loss.

`train_cpu_seconds` measures training; `total_cpu_seconds` also includes preparation, prediction, structure extraction and serialization. `model_*` columns summarize the selected ensemble. Full per-tree arrays are preserved in JSONL. Recorded model hashes describe original local UBJ files, which are intentionally not included here.

Each curve NPZ contains `train_loss`, `validation_loss`, cumulative `round_cpu`, cumulative `round_wall`, and `validation_prediction`. Loss/timing curves include all trained rounds; predictions use the best validation prefix. There are no test losses or test predictions. Test row indices are supplied only to reproduce the untouched split.

```python
import io, zipfile
import numpy as np

with zipfile.ZipFile("data/collection-2026-09-25/curves/openml_1464_r0.zip") as archive:
    raw = archive.read("openml_1464_r0__random_000.npz")
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        validation_curve = arrays["validation_loss"].copy()
```

## Provenance boundaries

OpenML IDs refer to fixed dataset versions. Exact row indices avoid recomputing version-sensitive random splits. Category vocabularies are fitted only on training rows; unknown validation categories become missing. Numeric infinities become missing. Binary/multiclass label ordering is archived.

Original data/model/curve hashes remain in provenance even when the corresponding source pickle or model is omitted. `manifest.json` hashes the portable files actually shipped; it does not claim original byte identity for transformed JSON/Parquet. Original NPZ curve bytes do match their original hashes.

The 50 families were selected from 120 historical-source representatives, excluding 59 reserved evaluation families. The earlier benchmark panel was already inspected; neither this collection nor the prior work should be presented as untouched confirmatory testing. Transient admission failures influenced the final roster. Reproduction fixes the admitted 50 IDs instead of repeating network-dependent admission.
