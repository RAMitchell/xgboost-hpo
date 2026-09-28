"""Regenerate mean validation regret and rank from the frozen trajectories.

Optional plotting dependency: pip install matplotlib.
"""

import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = Path(__file__).resolve().parent
with gzip.open(R / "trajectories.jsonl.gz", "rt") as f:
    rows = [json.loads(s) for s in f]
outcomes = json.loads((R / "outcomes.json").read_text())
methods = list(dict.fromkeys(r["method"] for r in rows))
families = sorted({r["uid"] for r in rows})
curves = np.empty((len(families), 10, len(methods), 32))
for row in rows:
    y = np.asarray(outcomes[row["uid"]]["objective"])
    denom = float(y.max() - y.min())
    curves[families.index(row["uid"]), row["rep"], methods.index(row["method"])] = (
        np.asarray(row["best"]) - y.min()
    ) / max(denom, 1e-12)
metrics = pd.read_csv(R / "metrics.csv").set_index("method")
for i, method in enumerate(methods):
    np.testing.assert_allclose(
        curves[:, :, i, :16].mean(), metrics.loc[method, "area1_16"], rtol=1e-12
    )
    np.testing.assert_allclose(
        curves[:, :, i, -1].mean(), metrics.loc[method, "regret32"], rtol=1e-12
    )
ranks = rankdata(curves, axis=2, method="average")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, data, title in zip(
    axes, [curves, ranks], ["Mean normalized regret", "Mean rank"]
):
    data = data.mean(axis=1)
    # Resample datasets as the independent unit, after averaging the ten seeds.
    boot = np.random.default_rng(9401).integers(
        len(families), size=(3000, len(families))
    )
    lo, hi = np.quantile(data[boot].mean(axis=1), [0.025, 0.975], axis=0)
    for i, name in enumerate(methods):
        line = ax.plot(np.arange(1, 33), data[:, i].mean(axis=0), label=name)[0]
        ax.fill_between(
            np.arange(1, 33), lo[i], hi[i], color=line.get_color(), alpha=0.12
        )
    ax.set(xlabel="Evaluations", ylabel=title)
    ax.grid(alpha=0.2)
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(R / "reconstructed.png", dpi=180)
print(
    "Saved benchmark/reconstructed.png; pointwise family-bootstrap intervals, not simultaneous bands."
)
