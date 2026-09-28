"""Verify the compact input snapshot and source/target separation; no training."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent


def main():
    manifest = json.loads((R / "data/manifest.json").read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((R / name).read_bytes()).hexdigest() == expected, name
    table = pd.read_parquet(R / "data/evaluations.parquet")
    datasets = json.loads((R / "datasets.json").read_text())
    targets = json.loads((R / "benchmark/datasets.json").read_text())
    configs = json.loads((R / "configurations.json").read_text())
    assert len(table) == 2000 and len(datasets) == 50 and len(configs) == 40
    assert table.status.value_counts().to_dict() == {"complete": 1999, "failed": 1}
    assert not table.duplicated(["uid", "cid"]).any()
    assert set(table.uid) == {d["uid"] for d in datasets}
    assert set(table.cid) == {c["cid"] for c in configs}
    assert not {d["family"] for d in datasets} & {d["family"] for d in targets}
    for d in datasets:
        with np.load(R / "splits" / (d["uid"] + ".npz"), allow_pickle=False) as a:
            # Original source-row indices, not feature tables.
            train = a["train"]
            validation = a["validation"]
            test = a["test"]
            assert (
                not set(train) & set(validation)
                and not set(train) & set(test)
                and not set(validation) & set(test)
            )
    ok = table.status == "complete"
    assert np.isfinite(table.loc[ok, "validation_loss"]).all()
    print(
        "Verified 50 source families, 2,000 records (one retained failure), splits, prior hashes and target exclusion."
    )


if __name__ == "__main__":
    main()
