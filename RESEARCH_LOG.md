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
