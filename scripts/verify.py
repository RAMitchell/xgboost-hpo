"""Verify checksums, coverage and curve/summary consistency without model training."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/collection-2026-09-25'


def main():
    manifest = json.loads((DATA/'manifest.json').read_text())['sha256']
    for name, expected in manifest.items():
        actual = hashlib.sha256((DATA/name).read_bytes()).hexdigest()
        if actual != expected: raise ValueError(f'Checksum mismatch: {name}')
    with gzip.open(DATA/'records.jsonl.gz', 'rt') as f:
        records = {z['record_id']: z for z in map(json.loads, f)}
    table = pd.read_parquet(DATA/'evaluations.parquet')
    datasets = json.loads((DATA/'datasets.json').read_text())
    configs = json.loads((DATA/'configurations.json').read_text())
    expected = {sp['uid']+'__'+c['cid'] for sp in datasets for c in configs}
    assert set(records) == set(table.record_id) == expected
    assert len(table) == len(expected) == 2000
    assert len({sp['family'] for sp in datasets}) == 50
    assert not {sp['family'] for sp in datasets} & set(json.loads((DATA/'reserved_families.json').read_text()))
    checked = 0
    for sp in datasets:
        with np.load(DATA/'splits'/(sp['uid']+'.npz'), allow_pickle=False) as splits:
            assert all(len(splits[n]) == sp['n_'+n] for n in ('train', 'validation', 'test'))
            rows = np.concatenate([splits[n] for n in ('train','validation','test')])
            assert len(np.unique(rows)) == len(rows) == sp['n_rows']
            assert rows.min() >= 0 and rows.max() < sp['original_rows']
        with zipfile.ZipFile(DATA/'curves'/(sp['uid']+'.zip')) as archive:
            valid = {rid+'.npz' for rid, z in records.items() if z['uid']==sp['uid'] and z['status']!='failed'}
            assert set(archive.namelist()) == valid
            for name in archive.namelist():
                z = records[name[:-4]]
                raw = archive.read(name)
                assert hashlib.sha256(raw).hexdigest() == z['curves_sha256']
                with np.load(io.BytesIO(raw), allow_pickle=False) as curve:
                    val = curve['validation_loss']
                    assert len(val) == len(curve['train_loss']) == z['trained_rounds']
                    assert np.isfinite(val).all() and np.isfinite(curve['train_loss']).all()
                    assert np.isfinite(curve['validation_prediction']).all()
                    assert val.argmin() == z['best_iteration']
                    assert np.isclose(val.min(), z['validation_loss'])
                row = table.loc[table.record_id == z['record_id']].iloc[0]
                assert row.validation_loss == z['validation_loss']
                assert json.loads((DATA/'configurations.json').read_text())[int(z['cid'].split('_')[1])]['params'] == z['params_sampled']
                checked += 1
    assert checked == 1999
    assert table.status.value_counts().to_dict() == {'complete':1999, 'failed':1}
    print(f'OK: {len(manifest)} checksums; 50 datasets; 2,000 records; 1,999 curves; one preserved failure.')


if __name__ == '__main__': main()
