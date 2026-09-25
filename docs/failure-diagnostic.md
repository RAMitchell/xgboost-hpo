# Collection failure diagnostic and actual dataset sizes

2026-09-25. Original collection artifacts are unchanged. One diagnostic objective-model training was run on physical core0 with the exact MiceProtein random_014 parameters, data, seed,2,000round ceiling,100round patience and120CPU-secondguard. Input data/build hashes verified.

The original failed line in database_collection_50/collector.py:81 combines four predicates:

```python
best == int(np.argmin(val))
len(val) == trained
np.isfinite(val).all()
np.isfinite(train).all()
```

The original failure record retained neither curves nor individual predicate values. Consequently we cannot identify its failed predicate conclusively from that record.

The rerun **passed all four checks**, completed2,000rounds in117.33CPU seconds, and never triggered theguard. Reportedbest iteration and validationargmin both1997; allcurve values finite. result.json anddiagnostic_curves.npz retain the evidence, anddiagnostic_model.ubj.zlib preserves the complete rerun model. This is a separate diagnostic, not an overwrite of the original failed record or silent modification of the frozen collection.

Strong suspected explanation: the original fit used120.28CPU seconds, near theguard threshold. In the frozen local XGBoost code, training appends EarlyStopping after user callbacks. CallbackContainer updates metric history, then uses short-circuit any(callback.after_iteration(...)). If Guard returnsTrue first, EarlyStopping does not see thefinal improved validation loss, leaving best_iteration stale while the curve already includes that round. That can violate the first predicate with otherwise valid model/losses. The rerun supports a timing-sensitive rather than deterministic numerical failure, but does not prove which predicate failed originally.

For a future collector version: report each invariant separately and preserve curves/guardstate even on failure; when stopped by the resourceguard, derive the best observed prefix from the finite validation curve and explicitly label the record censored. Do not silently increase thebudget or promote a truncated result to fullfidelity.

## Actual dataset sizes

50datasets: retainedrows528–20,000, median4,508. Trainingrowsmedian2,704,max12,080. Featuresmedian20, range3–4,991. Seventeen datasets have fewerretainedrows than their source; source rows were capped at20,000 and retainedfeaturecells at4million. Example: Santander_transaction_value4,991features×801retainedrows. See dataset_sizes.csv andsize_summary.json.

SuccessfulfitmediantrainingCPU0.759seconds,p90 7.722seconds,max49.319seconds. Mediantrainedrounds289 including early-stoppingpatience. End-to-end maincollection15.84minutes included newdata preparation/downloads; totaljobCPU1.85hours,averageabout7busycores overthe16-coreallocatedlimit. This is a bounded small/medium-data study, not evidence of transfer to unrestricted large datasets.
