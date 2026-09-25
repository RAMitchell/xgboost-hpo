# Reproducing the collection

For **new collections**, use `scripts/collect.py`; it reads the default 300 CPU-second training limit from `collection_defaults.json` and records that limit in run identity. Pass `--fit-cpu-seconds` only for an explicit override. The current replacement study has its own [overlay and replay instructions](../studies/expanded-prior-replacement-v1/README.md).

The historical `scripts/reproduce.py` and the design below retain the original 120-second limit for archival reproduction. Changing the current default does not rewrite those records.

## Two different guarantees

1. **Read and verify the archived results.** `python scripts/verify.py` verifies the portable manifest, 2,000-record coverage, disjoint row splits, family exclusion, and all 1,999 original curve hashes and loss summaries. This requires no training or OpenML access.
2. **Run the objective models again.** `python scripts/reproduce.py ...` fetches OpenML sources by ID, checks metadata, selects the saved rows, applies the recorded preprocessing, and trains the saved parameter dictionaries. Results depend on the installed XGBoost build and timing caps.

The first snapshot used a 3.5.0-dev shared library with SHA-256 `b37cfda4bc173750622ae62c4731588fa206dd055ab47ea26bcf2582becdf089`. The source snapshot reported revision `6f76db0be60599f9f91393e5291daa09d076468c`, but the shared library was copied from an existing build, not rebuilt with independently verified source ancestry. We therefore do not claim this revision alone can recreate the original binary. The binary is not bundled.

`provenance/python_packages.json` records available dependency versions; `requirements.txt` lists practical compatible dependencies, not an exact historical lock. A future confirmatory collection should use a released, pinned XGBoost wheel or a clean pinned build with a recorded toolchain. The portable runner records its actual installed library hash, package versions, and script hashes and prevents mixed-environment resumes.

## Training design

- CPU histogram trees; one XGBoost/BLAS thread per worker.
- 2,000 boosting-round ceiling; 100 rounds of validation early-stopping patience.
- 120 process-CPU-second guard per fit; the guard can overshoot by one boosting iteration.
- Binary/multiclass log loss or regression RMSE, minimized.
- Fixed dataset seed `20260924 + OpenML_ID`.
- Train-only categorical vocabulary and validation QuantileDMatrix referencing train.
- No test loss evaluation.

The original 16-core run took 950.4 seconds end-to-end and 6,661.5 job CPU seconds. Median successful training CPU time was 0.759 seconds. Downloads, memory, machine load, XGBoost version, and the wide-dataset tail affect replication time. The portable runner uses at most the requested number of one-thread workers but does not pin physical cores; set OS affinity externally if required.

## Collector changes made for portability

The original scripts are archived under `provenance/original_scripts/` as historical evidence. They depend on their original study layout and are not the supported entry point.

The supported `scripts/` runner:

- Uses installed XGBoost, relative repository paths, and an explicit output directory.
- Fixes the admitted roster, fetches by OpenML ID, and restores saved source-row splits rather than rerunning admission or random split generation.
- Stores reconstructed frames in Parquet instead of Python pickle.
- Derives the selected prefix from the finite validation curve after a resource guard stop, marking it **censored**; checks the reported best iteration on unguarded runs.
- Saves available histories and guard state when training fails.
- Enforces a single controller per output directory with an advisory lock.

The guard change addresses a suspected stale-best-iteration issue, not a conclusively reproduced original failure. The one original failed record remains failed in the public snapshot. A separate diagnostic rerun passed 2,000 rounds in 117.3 CPU seconds; its metadata is under provenance and it is not counted among the 1,999 successful collection records.

To change parameters or environment, use a new output directory. To stop, create `<output>/STOP`; remove it to resume. Existing failed or censored records stay visible and are reused rather than automatically replaced. A nonzero exit status indicates missing, failed, or censored requested evaluations. Do not interpret successful completion at the round ceiling as asymptotic convergence.
