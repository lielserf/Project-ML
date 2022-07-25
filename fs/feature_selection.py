import numpy as np
import time
from fs.MRMR import mrmr
from fs.reliefF import reliefF, feature_ranking
from sklearn.feature_selection import SelectFdr, f_classif, RFE
from sklearn.svm import SVC
from fs.DF import ReduceDF


def run_reducer(method, X, y):
    reducer_feature = dict()
    if method == 'mRmr':
        start = time.time()
        features, scores, _ = mrmr(X, y, n_selected_features=100)
        run_time = time.time() - start
    elif method == 'ReliefF':
        start = time.time()
        score = reliefF(X, y)
        run_time = time.time() - start
        scores = np.sort(score)[:100:]
        features = feature_ranking(score)[:100]
    elif method == 'RFE':
        Rfe = RFE(SVC(kernel="linear"), n_features_to_select=1)
        start = time.time()
        Rfe.fit(X, y)
        timeit = time.time() - start
        features, scores = get_features_and_scorer('Rfe', Rfe)
        run_time = timeit
    elif method == 'Fdr':
        Fdr = SelectFdr(f_classif, alpha=0.1)
        start = time.time()
        Fdr.fit(X, y)
        timeit = time.time() - start
        features, scores = get_features_and_scorer('Fdr', Fdr)
        run_time = timeit
    elif method == 'DF':
        DF = ReduceDF(n_features_to_select=100)
        start = time.time()
        DF.fit(X, y)
        timeit = time.time() - start
        # features, scores = get_features_and_scorer('DF', DF)
        # run_time = timeit
    reducer_feature['features'] = features.tolist()
    reducer_feature['scorer'] = scores.tolist()
    reducer_feature['time'] = run_time
    return reducer_feature


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
    elif reducer == 'DF':
        features = reducer_method.features[:100]
        score = reducer_method.ranking[:100]
        return features, score



