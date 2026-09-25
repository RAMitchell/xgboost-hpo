"""Load results and a loss curve, without downloading OpenML datasets."""
import io
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data/collection-2026-09-25'
results = pd.read_parquet(DATA / 'evaluations.parquet')
complete = results[results.status == 'complete']
print(complete.groupby('problem_type').agg(datasets=('uid','nunique'), evaluations=('cid','size')))
row = complete.iloc[0]
with zipfile.ZipFile(DATA/'curves'/f'{row.uid}.zip') as archive:
    with np.load(io.BytesIO(archive.read(row.record_id+'.npz')), allow_pickle=False) as curve:
        print(row.record_id, 'best validation loss:', curve['validation_loss'].min())
