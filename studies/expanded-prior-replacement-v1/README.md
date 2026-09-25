# Five-minute evaluation with Dilbert replaced

This follows the user's request to exclude Dilbert because of excessive training time. The first eligible unused multiclass dataset in the original reserved order replaces it. The historical prior, other 29 datasets, 96 configurations, seven optimizer methods and seeds remain fixed. The timeout is 300 training CPU-seconds; execution uses up to 32 physical cores.

See [PROTOCOL.md](PROTOCOL.md), [reports/RESULTS.md](reports/RESULTS.md), and [reports/regret_and_rank.png](reports/regret_and_rank.png). This is a post-hoc change to the benchmark population, and must be described as such when interpreting improvement over the original results.

## Data overlay

`../../data/heldout-expanded-replacement-v1` overlays `../../data/heldout-expanded-v1`:

1. Remove records for `openml_41163_r0` (Dilbert).
2. Replace `openml_12_r0__eval_011` with its completed five-minute rerun.
3. Add the replacement dataset's 96 records.
4. Use the new `datasets.json` as the authoritative roster. Retain parent splits for existing datasets and use the new split archive for the replacement.

All source datasets remain represented by OpenML IDs, versions and exact source-row splits. Raw input tables and trained model binaries are not redistributed.

## Reproduction

```sh
python studies/expanded-prior-replacement-v1/verify.py
python studies/expanded-prior-replacement-v1/replay_published.py --workers 32 --output runs/replay-replacement
```

These commands verify or replay the archived data without objective training. Dependencies for replay are pinned in `requirements.txt`. The learned prior is unchanged from `../expanded-prior-v1`.

For new objective collection, use `scripts/collect.py`; its default timeout is 300 seconds and is part of the run identity. The local incremental controller, `run.py`, assumes the parent prepared run and completed mfeat-factors rerun are available. It downloads and prepares only the replacement dataset, collects 96 fits and automatically generates the new comparison. See `runs/expanded-prior-replacement-v1` for local launch metadata, logs and progress.

To reconstruct and retrain the newly collected configurations from OpenML:

```sh
python scripts/collect.py --snapshot data/heldout-expanded-replacement-v1 --dataset-id 40985 --workers 32 --output runs/replacement-reproduction
python scripts/collect.py --snapshot data/heldout-expanded-v1 --dataset-id 12 --config-id eval_011 --workers 32 --output runs/mfeat-reproduction
```

The new snapshot is an overlay, so its split directory contains only the new dataset. Use the parent snapshot when retraining a retained dataset; do not pass `--all` to the overlay. The remaining retained evaluations are reproduced with the parent instructions. Exact numerical reproduction requires the recorded development binary; a released XGBoost version may differ.
