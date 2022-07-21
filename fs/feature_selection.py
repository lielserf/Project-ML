import numpy as np


def get_features_and_scorer(reducer, reducer_method):
    if reducer == 'Fdr':
        p_v = reducer_method.pvalues_
        features = np.argsort(p_v)[:100]
        score = np.sort(p_v)[:100]
        return features, score
    elif reducer == 'Rfe':
        ranking = reducer_method.ranking_
        features = np.argsort(ranking)[:100]
        score = np.sort(ranking)[:100]
        return features, score



