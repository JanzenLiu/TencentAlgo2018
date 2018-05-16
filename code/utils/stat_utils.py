import numpy as np
from scipy.stats import logistic


def inverse_sigmoid(y):
    return np.log( y / (1 - y))


def merge_proba(*probas):
    num_probas = len(probas)
    large_proba = np.zeros((probas[0].shape[0], num_probas))

    for i in range(num_probas):
        large_proba[:, i] = inverse_sigmoid(probas[i])

    large_proba = large_proba.mean(axis=1)
    large_proba = logistic.cdf(large_proba)

    return large_proba
