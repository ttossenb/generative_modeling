# sklearn version. see kl_mixture_training.py for
# a version that can optimize KL of mixtures.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture as sklearn_mixture

k = 1000
d = 10

def create_mixture(means, variances, weights=None):
    k, d = means.shape
    assert means.shape == variances.shape
    if weights is not None:
        assert len(weights) == k
    clf = sklearn_mixture.GaussianMixture(n_components=k, covariance_type='diag')
    # just so that it creates all members. class not designed for anything other than fitting.
    clf.fit(np.random.normal(size=(2*k, d)))
    if weights is not None:
        clf.weights_ = weights
    else:
        clf.weights_ = np.ones_like(clf.weights_) / k
    clf.means_ = means
    clf.covariances_ = variances
    clf.precisions_ = 1 / clf.covariances_
    clf.precisions_cholesky_ = np.sqrt(clf.precisions_)
    return clf


means = np.random.normal(size=(k, d))
variances = np.random.uniform(size=(k, d)) / 100
mixture = create_mixture(means, variances)

# print(mixture.score_samples(np.random.normal(size=(10, d))))

def kl(mixture, log_density, nr_samples):
    samples, components = mixture.sample(nr_samples)
    log_probs = mixture.score_samples(samples)
    kl_div = np.mean(log_probs - log_density(samples))
    return kl_div

def own_log_density(samples):
    return mixture.score_samples(samples)

def prior_log_density(samples):
    n, d = samples.shape
    sq = np.linalg.norm(samples, axis=1) ** 2
    log_probs = - sq / 2 - d / 2 * np.log(2*np.pi)
    return log_probs


degen_mixture = create_mixture(np.zeros((1, d)), np.ones((1, d)))

print(kl(mixture, own_log_density, 10000))
print(kl(mixture, prior_log_density, 10000))
print(kl(degen_mixture, prior_log_density, 10000))

print("======")

for variance in np.linspace(0.01, 2, 20):
    mixture = create_mixture(np.array([-1, +1]).reshape((2, 1)), variance * np.ones((2, 1)))
    print(variance, kl(mixture, prior_log_density, 10000))

print("======")

print("k =", k, "d =", d)
noise = np.random.normal(size=(k, d))
for variance in np.linspace(0, 2.0, 20)[1:]:
    mixture = create_mixture(noise, variance * np.ones((k, d)))
    print(variance, kl(mixture, prior_log_density, 100000))
