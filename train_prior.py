"""Fit mean, scale and ARD kernel exclusively from the 50 source families."""

import os

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize

from prior_utils import FEATURES, encode, kernel, normal_rank_scores, objective
from verify import main as verify_inputs

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT


def read(path):
    return json.loads(Path(path).read_text())


def write(p, v):
    Path(p).write_text(json.dumps(v, indent=2) + "\n")


def fit(output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    start = time.process_time()
    wall = time.perf_counter()
    table = pd.read_parquet(SOURCE / "data/evaluations.parquet")
    configs = read(SOURCE / "configurations.json")
    X = encode(configs)
    ds = read(SOURCE / "datasets.json")
    mass = np.zeros(len(X))
    target = np.zeros(len(X))
    tasks = []
    for sp in ds:
        rows = table[
            (table.uid == sp["uid"]) & (table.status == "complete")
        ].sort_values("cid")
        idx = np.array([int(cid.split("_")[-1]) for cid in rows.cid])
        y = normal_rank_scores(rows.validation_loss.to_numpy())
        w = 1 / len(ds)
        mass[idx] += w
        target[idx] += w * y
        tasks.append((idx, y, w))
    target /= mass
    weight = mass / mass.sum() * 200
    params = dict(
        objective="reg:squarederror",
        max_depth=3,
        eta=0.05,
        tree_method="hist",
        max_bin=128,
        nthread=1,
        min_child_weight=1,
        reg_lambda=1,
        reg_alpha=0,
        gamma=0,
        subsample=1.0,
        colsample_bytree=1.0,
    )
    dm = xgb.DMatrix(X, label=target, weight=weight, nthread=1)
    mean_model = xgb.train(
        dict(params, base_score=0.0, seed=9025), dm, num_boost_round=200
    )
    m = mean_model.predict(dm).astype(float)
    mean_model.save_model(output / "mean.ubj")
    second = np.zeros(len(X))
    for idx, y, w in tasks:
        second[idx] += w * (y - m[idx]) ** 2
    logs = np.log(np.maximum(0.0025, second / mass))
    dm.set_label(logs)
    scale_model = xgb.train(
        dict(params, base_score=float(np.average(logs, weights=weight)), seed=9026),
        dm,
        num_boost_round=200,
    )
    s = np.maximum(
        0.05, np.exp(np.clip(scale_model.predict(dm).astype(float), -12, 12) / 2)
    )
    scale_model.save_model(output / "scale.ubj")
    buckets = {}
    for idx, y, w in tasks:
        k = tuple(idx)
        r = (y - m[idx]) / s[idx]
        if k not in buckets:
            buckets[k] = [np.zeros((len(idx), len(idx))), 0.0]
        buckets[k][0] += w * np.outer(r, r)
        buckets[k][1] += w
    groups = []
    for ids, (mom, w) in buckets.items():
        xx = X[list(ids)]
        groups.append(((xx[:, None] - xx[None, :]) ** 2, mom / w, w))
    bounds = np.log([[0.01, 100.0]] * 18 + [[1e-3, 1e3], [1e-6, 1.0]])
    traces = []
    for ell, noise in [(0.2, 1e-6), (0.5, 0.01), (1.0, 0.1)]:
        opt = minimize(
            lambda th: objective(th, groups),
            np.log([ell] * 18 + [1.0, noise]),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=300, ftol=1e-10, gtol=1e-6),
        )
        traces.append(
            dict(
                value=float(opt.fun),
                theta=opt.x.tolist(),
                success=bool(opt.success),
                message=str(opt.message),
                iterations=int(opt.nit),
            )
        )
    converged = [z for z in traces if z["success"]]
    assert converged, "No converged historical kernel fit"
    theta = np.array(min(converged, key=lambda z: z["value"])["theta"])
    Q = encode([{"params": p} for p in read(ROOT / "candidates.json")])
    qd = xgb.DMatrix(Q, nthread=1)
    mean = mean_model.predict(qd).astype(float)
    scale = np.maximum(
        0.05, np.exp(np.clip(scale_model.predict(qd).astype(float), -12, 12) / 2)
    )
    K = kernel(theta, (Q[:, None] - Q[None, :]) ** 2) * scale[:, None] * scale[
        None, :
    ] + 1e-10 * np.eye(len(Q))
    noise = scale**2 * np.exp(theta[-1])
    np.linalg.cholesky(K)
    np.savez_compressed(
        output / "candidate_prior.npz", mean=mean, covariance=K, noise_variance=noise
    )
    model = dict(
        source_families=sorted(sp["family"] for sp in ds),
        source_datasets=len(ds),
        source_observations=sum(len(y) for _, y, _ in tasks),
        distinct_configurations=len(X),
        features=FEATURES,
        mean_training_predictions=m.tolist(),
        scale_training_predictions=s.tolist(),
        logtheta=theta.tolist(),
        lengthscales=dict(zip(FEATURES, np.exp(theta[:18]).tolist())),
        amplitude=float(np.exp(theta[-2])),
        noise_variance=float(np.exp(theta[-1])),
        restarts=traces,
        boundaries=np.flatnonzero(
            np.any(np.isclose(theta[:, None], bounds, atol=1e-4), axis=1)
        ).tolist(),
        parameters=params,
        mean_scale_boosting_rounds=200,
        total_fit_weight=200,
        cpu_seconds=time.process_time() - start,
        wall_seconds=time.perf_counter() - wall,
        historical_xgb_fits=2,
        kernel_starts=3,
        target_losses_read=0,
    )
    write(output / "model.json", model)
    print(
        json.dumps(
            {
                k: model[k]
                for k in [
                    "source_observations",
                    "distinct_configurations",
                    "cpu_seconds",
                    "boundaries",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/prior")
    args = parser.parse_args()
    verify_inputs()
    fit(args.output)
    manifest = read(ROOT / "prior/manifest.json")
    manifest["files"] = {
        n: hashlib.sha256((args.output / n).read_bytes()).hexdigest()
        for n in ["mean.ubj", "scale.ubj", "model.json"]
    }
    manifest["reproduction_xgboost_version"] = xgb.__version__
    manifest["reproduction_library_sha256"] = hashlib.sha256(
        Path(xgb.core._LIB._name).read_bytes()
    ).hexdigest()
    import scipy

    manifest["reproduction_numpy_version"] = np.__version__
    manifest["reproduction_scipy_version"] = scipy.__version__
    manifest["reproduction_script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    manifest["source_manifest_sha256"] = hashlib.sha256(
        (ROOT / "data/manifest.json").read_bytes()
    ).hexdigest()
    write(args.output / "manifest.json", manifest)
