# Examples

`read_database.py` reads the packaged Parquet results and an original loss curve without downloading source datasets.

`quantile_gp_numpy.py` is the NumPy-only extraction of the earlier two-start quantile-GP online algorithm. It exposes `QuantileGP(mean, covariance, noise_variance, seed=0)`, `ask()`, `tell(candidate_id, loss)`, and `best_observed()`. Loss is minimized. The caller provides mean/covariance on a fixed candidate pool; noise variance can be scalar or one value per candidate. Two prior Thompson proposals precede rank-normalized expected improvement.

`ask_tell.py` illustrates the interface with a toy, manually specified smooth prior and synthetic objective. It is not a benchmark, a historical prior, or a new default recommendation. Prior training, kernel learning and conversion from hyperparameters to a covariance matrix are separate responsibilities.
