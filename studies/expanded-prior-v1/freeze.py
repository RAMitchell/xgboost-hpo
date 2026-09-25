from common import *
import xgboost as xgb
import importlib.metadata as md
import platform

def main():
 assert not (E/'freeze.json').exists()
 cpus=physical_cores();assert len(cpus)==16
 write(E/'design.json',dict(target_datasets=30,candidates=96,budget=32,optimizer_seeds=list(range(10)),methods=METHODS,source_sha256=digest(SOURCE/'manifest.json'),cpus=cpus,workers=16,threads_per_fit=1))
 write(E/'environment.json',dict(python=platform.python_version(),packages={p:md.version(p) for p in ['numpy','pandas','scipy','scikit-learn','optuna','smac','ConfigSpace','matplotlib','pyarrow']},xgboost=xgb.__version__,library_sha256=digest(xgb.core._LIB._name)))
 paths=list(E.glob('*.py'))+list(E.glob('*.json'))+list((E/'prior').glob('*'))+[E/'PROTOCOL.md',ROOT/'scripts/collector.py',ROOT/'scripts/configuration_mapping.py',ROOT/'examples/quantile_gp_numpy.py',SOURCE/'manifest.json',SOURCE/'evaluations.parquet',SOURCE/'datasets.json',SOURCE/'configurations.json']
 hashes={str(p.relative_to(ROOT)):digest(p) for p in paths if p.is_file()};write(E/'freeze.json',dict(identity=key(hashes),hashes=hashes,target_outcomes_read=0));print('Frozen',key(hashes))
if __name__=='__main__':main()
