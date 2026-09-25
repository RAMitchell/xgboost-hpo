"""Project the frozen historical prior onto a legal new candidate pool."""
from common import *
import xgboost as xgb


def predict_prior(configurations):
    """Return mean, latent covariance and observation-noise variance.

    Each configuration is a {'params': ...} dictionary in the archived design.
    No target losses or dataset metadata are consumed.
    """
    for config in configurations:
        p = config['params']
        if p['grow_policy'] not in ('depthwise', 'lossguide'):
            raise ValueError('Unknown grow policy')
        if p['max_depth'] != int(p['max_depth']) or p['max_leaves'] != int(p['max_leaves']):
            raise ValueError('Tree size bounds must be integers')
        if p['max_depth'] == 0 and p['max_leaves'] == 0:
            raise ValueError('A depth or leaf limit is required')
        if p['max_bin'] not in (128, 256, 512):
            raise ValueError('Unsupported max_bin')
        if sum(p['colsample_by'+m] < 1 for m in ('tree','level','node')) > 1:
            raise ValueError('Historical design supports only one active column sampling mode')
    X = encode(configurations)
    models = []
    for name in ('mean','scale'):
        model = xgb.Booster(params={'nthread':1})
        model.load_model(E/'prior'/f'{name}.ubj')
        models.append(model)
    data = xgb.DMatrix(X,nthread=1)
    mean = models[0].predict(data).astype(float)
    scale = np.maximum(.05,np.exp(np.clip(models[1].predict(data).astype(float),-12,12)/2))
    theta = np.array(read(E/'prior/model.json')['logtheta'])
    covariance = kernel(theta,(X[:,None]-X[None,:])**2)*scale[:,None]*scale[None,:]+1e-10*np.eye(len(X))
    return mean,covariance,scale**2*np.exp(theta[-1])


if __name__ == '__main__':
    mean,covariance,noise=predict_prior(read(E/'configurations.json'))
    with np.load(E/'prior/candidate_prior.npz') as saved:
        for key,value in [('mean',mean),('covariance',covariance),('noise_variance',noise)]:
            np.testing.assert_array_equal(value,saved[key])
    print('Prior projection exactly matches the frozen evaluation prior.')
