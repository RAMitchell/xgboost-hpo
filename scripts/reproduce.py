"""Reconstruct the archived OpenML splits and rerun selected configurations.

Results go into a separate run directory. The published database is immutable.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/collection-2026-09-25'


def prepare(sp, output, cache):
    import numpy as np
    import pandas as pd
    from sklearn.datasets import fetch_openml
    from collector import write, digest

    directory = output / 'data' / sp['uid']
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / 'receipt.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        for name, expected in saved['files'].items():
            if digest(directory / name) != expected:
                raise ValueError(f'Prepared data changed: {directory / name}')
        if saved['source_identity'] != sp['data_identity']:
            raise ValueError('Prepared data belongs to another source snapshot')
        return dict(sp, data_dir=str(directory))

    data = fetch_openml(data_id=sp['did'], as_frame=True, parser='auto',
                        data_home=cache, n_retries=3, delay=2.)
    X, target = data.data.copy(), data.target
    if list(X.columns) != sp['features'] or str(target.name) != sp['target_name']:
        raise ValueError('OpenML feature/target schema differs from the snapshot')
    if str(data.details.get('version')) != str(sp['openml_version']):
        raise ValueError('OpenML dataset version changed')
    if sp.get('source_md5') and data.details.get('md5_checksum') != sp['source_md5']:
        raise ValueError('OpenML source checksum changed')
    good = ~target.isna()
    if sp['task'] == 'regression':
        y = pd.to_numeric(target, errors='raise').to_numpy(float)
    else:
        _, labels = pd.factorize(target.loc[good], sort=True)
        if list(map(str, labels)) != sp['classes']:
            raise ValueError('Class ordering differs from the snapshot')
        y = pd.Categorical(target, categories=labels).codes
    for col in sp['features']:
        if col in sp['categoricals']:
            X[col] = X[col].astype('string')
        else:
            X[col] = pd.to_numeric(X[col], errors='raise').astype(float)
            X.loc[np.isinf(X[col]), col] = np.nan
    with np.load(DATA / 'splits' / (sp['uid'] + '.npz'), allow_pickle=False) as splits:
        all_rows = np.concatenate([splits[n] for n in ('train', 'validation', 'test')])
        if len(np.unique(all_rows)) != sp['n_rows']:
            raise ValueError('Split indices overlap or do not cover retained rows')
        for name in ('train', 'validation', 'test'):
            rows = splits[name]
            if not good.iloc[rows].all() or not np.isfinite(y[rows]).all():
                raise ValueError('Invalid targets in retained rows')
            # Preserve original source indices; collection used iloc with this exact order.
            X.iloc[rows].reset_index(drop=True).to_parquet(directory / f'{name}_X.parquet', index=False, compression='zstd')
            np.save(directory / f'{name}_y.npy', y[rows])
    files = {p.name: digest(p) for p in directory.iterdir() if p.suffix in ('.parquet', '.npy')}
    write(receipt, dict(source_identity=sp['data_identity'], files=files))
    return dict(sp, data_dir=str(directory))


def worker(job):
    from collector import evaluate
    sp, config, identity = job
    return evaluate(sp, config, identity)


def main():
    global DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=DATA, help='Packaged snapshot directory')
    parser.add_argument('--dataset-id', type=int, action='append', help='Repeat for multiple OpenML IDs')
    parser.add_argument('--all', action='store_true', help='Explicitly rerun every dataset in the snapshot')
    parser.add_argument('--config-id', action='append', help='Default: all configurations in the snapshot')
    parser.add_argument('--workers', type=int, default=1, help='Concurrent one-thread fits')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/reproduction')
    parser.add_argument('--cache', type=Path, default=ROOT / 'runs/openml-cache')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    DATA = args.snapshot.resolve()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if bool(args.dataset_id) == args.all:
        parser.error('Choose --dataset-id or --all')
    output = args.output.resolve()
    if output == DATA or DATA in output.parents:
        parser.error('Output must be outside the published data directory')
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / '.collection.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['HPO_RUN_DIR'] = str(output)
    from collector import write, key
    import xgboost as xgb
    import numpy, pandas, scipy, sklearn
    datasets = json.loads((DATA / 'datasets.json').read_text())
    configs = json.loads((DATA / 'configurations.json').read_text())
    if args.dataset_id:
        missing = set(args.dataset_id) - {sp['did'] for sp in datasets}
        if missing: parser.error(f'Unknown dataset IDs: {missing}')
        datasets = [sp for sp in datasets if sp['did'] in args.dataset_id]
    if args.config_id:
        missing = set(args.config_id) - {c['cid'] for c in configs}
        if missing: parser.error(f'Unknown configuration IDs: {missing}')
        configs = [c for c in configs if c['cid'] in args.config_id]
    environment = dict(xgboost=xgb.__version__, numpy=numpy.__version__, pandas=pandas.__version__,
                       scipy=scipy.__version__, sklearn=sklearn.__version__,
                       library_sha256=hashlib.sha256(Path(xgb.core._LIB._name).read_bytes()).hexdigest(),
                       scripts={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'scripts').glob('*.py'))})
    identity = key(environment)
    env_file = output / 'environment.json'
    if env_file.exists() and json.loads(env_file.read_text()) != environment:
        raise ValueError('Environment/code changed; choose a new output directory')
    write(env_file, environment)
    (output / 'records').mkdir(exist_ok=True)
    results = []
    # Prepare one dataset at a time; avoid loading the entire collection into memory.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        for sp in datasets:
            if (output / 'STOP').exists(): break
            prepared = prepare(sp, output, args.cache)
            print(f"Prepared {sp['dataset']} ({sp['did']})", flush=True)
            if args.prepare_only: continue
            jobs = [(prepared, c, key([identity, sp['data_identity']])) for c in configs]
            for z in pool.map(worker, jobs, chunksize=1):
                results.append(z)
                print(f"{z['record_id']}: {z['status']} loss={z.get('validation_loss')}", flush=True)
    if not args.prepare_only:
        counts = {status: sum(z['status'] == status for z in results) for status in ('complete', 'censored', 'failed')}
        write(output/'summary.json', dict(counts=counts, expected=len(datasets)*len(configs), records=len(results)))
        if counts['failed'] or counts['censored'] or len(results) != len(datasets)*len(configs):
            raise SystemExit('Run contains failed/censored/missing evaluations; inspect summary.json')


if __name__ == '__main__':
    main()
