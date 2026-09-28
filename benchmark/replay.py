"""Replay the 300 frozen prior-GP runs with the experimental XGBoost branch."""

import gzip
import json
from pathlib import Path

import numpy as np
from xgboost import hpo

R = Path(__file__).resolve().parents[1]
candidates = json.loads((R / "candidates.json").read_text())
prior = hpo.Prior.load(R / "prior", candidates=candidates)
outcomes = json.loads((R / "benchmark/outcomes.json").read_text())
count = 0
with gzip.open(R / "benchmark/trajectories.jsonl.gz", "rt") as stream:
    for line in stream:
        record = json.loads(line)
        if record["method"] != "gp_prior":
            continue
        optimizer = hpo.Optimizer(candidates, prior=prior, random_state=record["seed"])
        for expected, value in zip(record["ids"], record["values"]):
            trial = optimizer.ask()
            assert trial.candidate_id == expected, (
                record["uid"],
                record["rep"],
                trial.number,
                trial.candidate_id,
                expected,
            )
            assert np.isclose(
                value, outcomes[record["uid"]]["objective"][expected], rtol=0, atol=0
            )
            optimizer.tell(trial, value)
        count += 1
assert count == 300, count
print(
    f"Reproduced {count} paths / {count*32} sequential proposals; zero objective fits."
)
