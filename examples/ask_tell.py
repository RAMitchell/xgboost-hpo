"""Illustrate the GP API on a toy candidate pool; this is not a learned prior."""
import numpy as np
from quantile_gp_numpy import QuantileGP

x = np.linspace(0., 1., 40)
covariance = np.exp(-0.5*((x[:,None]-x[None,:])/0.2)**2) + 1e-10*np.eye(len(x))
optimizer = QuantileGP(mean=np.zeros(len(x)), covariance=covariance, noise_variance=0.01, seed=0)
for _ in range(10):
    candidate_id = optimizer.ask()
    loss = (x[candidate_id]-0.37)**2
    optimizer.tell(candidate_id, loss)
print('Best candidate and loss:', optimizer.best_observed())
