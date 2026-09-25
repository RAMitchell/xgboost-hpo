"""NumPy-only-dependency sequential extraction of the evaluated two-start quantile GP.

The caller supplies the historical prior on a fixed candidate pool. Lower loss
is better. No kernel fitting, historical training, persistence, or networking.
"""
import hashlib
import json
import math
import numpy as np
from statistics import NormalDist


def normal_rank_scores(y):
    """Normal scores of average ranks; ties receive the same score.

    NormalDist is part of Python's standard library. NumPy is the only
    third-party dependency, including for the GP linear algebra.
    """
    _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    quantiles = ((np.cumsum(counts) - 0.5 * counts) / len(y))[inverse]
    inverse_cdf = NormalDist().inv_cdf
    return np.fromiter((inverse_cdf(float(p)) for p in quantiles), float, len(y))


def _seed(*items):
    return int.from_bytes(
        hashlib.sha256(json.dumps(items).encode()).digest()[:4], 'little'
    )


def log_ei(mean, sd, best):
    """Stable log expected improvement for minimization; frozen study formula."""
    mean, sd = np.asarray(mean, float), np.asarray(sd, float)
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(sd)) or np.any(sd < 0):
        raise ValueError('Invalid posterior moments')
    out = np.full(mean.shape, -np.inf)
    positive = sd > 0
    z = (best - mean[positive]) / sd[positive]
    v = np.empty_like(z)
    central = z >= -1
    middle = (z >= -20) & ~central
    tail = z < -20
    x = z[central]
    cdf = np.fromiter(
        (0.5 * math.erfc(-float(t) / math.sqrt(2)) for t in x), float, len(x)
    )
    v[central] = np.log(np.exp(-0.5 * x*x) / math.sqrt(2*math.pi) + x*cdf)
    x = z[middle]
    scaled_erfc = np.fromiter(
        (math.exp(float(t*t)/2) * math.erfc(-float(t)/math.sqrt(2)) for t in x),
        float, len(x),
    )
    v[middle] = (-0.5*x*x - 0.5*math.log(2*math.pi)
                 + np.log1p(x * math.sqrt(math.pi/2) * scaled_erfc))
    x = z[tail]
    u = 1 / (x*x)
    series, term = np.ones_like(x), np.ones_like(x)
    for k in range(1, 11):
        term *= -(2*k + 1) * u
        series += term
    v[tail] = (-0.5*x*x - 0.5*math.log(2*math.pi)
               - 2*np.log(-x) + np.log(series))
    out[positive] = np.log(sd[positive]) + v
    deterministic = ~positive & (best > mean)
    out[deterministic] = np.log(best - mean[deterministic])
    return out


class QuantileGP:
    def __init__(self, mean, covariance, noise_variance, seed=0):
        self.mean = np.asarray(mean, dtype=float).copy()
        self.K = np.asarray(covariance, dtype=float).copy()
        n = len(self.mean)
        self.noise = np.broadcast_to(noise_variance, (n,)).astype(float).copy()
        if self.mean.shape != (n,) or self.K.shape != (n, n) or n < 2:
            raise ValueError('Expected a mean vector and matching covariance')
        if not all(np.isfinite(a).all() for a in (self.mean, self.K, self.noise)):
            raise ValueError('Prior inputs must be finite')
        if np.any(self.noise < 0) or not np.allclose(self.K, self.K.T):
            raise ValueError('Invalid noise variance or asymmetric covariance')
        # Covariance already includes the study's 1e-10 diagonal jitter.
        self.prior_cholesky = np.linalg.cholesky(self.K)
        self.seed = int(seed)
        self.start_rng = np.random.default_rng(
            _seed('fewshot-prior-start-v1', self.seed)
        )
        self.seen, self.losses = [], []
        self.pending = None

    def _posterior(self):
        ids = np.asarray(self.seen)
        y = np.asarray(self.losses)
        best = ids[np.argmin(y)]
        mean = self.mean.copy()
        # Variance of f(best) - f(x), including their covariance.
        difference_var = self.K[best, best] + np.diag(self.K) - 2*self.K[:, best]
        if np.ptp(y) != 0:
            scores = normal_rank_scores(y)
            observed_cov = self.K[np.ix_(ids, ids)] + np.diag(self.noise[ids])
            L = np.linalg.cholesky(observed_cov)
            cross = self.K[:, ids]
            # Solve all right-hand sides together; never form an inverse.
            rhs = np.column_stack((scores - self.mean[ids], cross.T))
            solved = np.linalg.solve(L, rhs)
            residual, V = solved[:, 0], solved[:, 1:]
            mean += V.T @ residual
            difference_V = V[:, best, None] - V
            difference_var -= np.sum(difference_V**2, axis=0)
        tolerance = 1e-7 * max(1.0, float(np.diag(self.K).max()))
        if difference_var.min() < -tolerance:
            raise ArithmeticError('Materially negative posterior variance')
        return mean[best] - mean, np.maximum(difference_var, 0)

    def ask(self):
        if self.pending is not None:
            return self.pending  # Sequential API: wait for tell().
        available = np.setdiff1d(np.arange(len(self.mean)), self.seen)
        if not len(available):
            raise StopIteration('Candidate pool exhausted')
        if len(self.seen) < 2:
            draw = self.mean + self.prior_cholesky @ self.start_rng.standard_normal(
                len(self.mean)
            )
            cid = available[np.argmin(draw[available])]
        else:
            improvement_mean, improvement_var = self._posterior()
            scores = log_ei(
                -improvement_mean[available],
                np.sqrt(improvement_var[available]),
                best=0.0,
            )
            rng = np.random.default_rng(
                _seed('fewshot-choice-v1', self.seed, len(self.seen))
            )
            cid = rng.choice(available[scores == scores.max()])
        self.pending = int(cid)
        return self.pending

    def tell(self, candidate_id, loss):
        if self.pending is None or candidate_id != self.pending:
            raise ValueError('tell() must match the outstanding ask()')
        if not np.isfinite(loss):
            raise ValueError('Loss must be finite')
        self.seen.append(self.pending)
        self.losses.append(float(loss))
        self.pending = None

    def best_observed(self):
        if not self.seen:
            raise ValueError('No evaluations completed')
        i = int(np.argmin(self.losses))
        return self.seen[i], self.losses[i]
