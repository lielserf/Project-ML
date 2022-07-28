from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif as MIC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from pyitlib import discrete_random_variable as drv


class ReduceDF():

    def __init__(
            self,
            n_features_to_select,

    ):
        self.n_features_to_select = n_features_to_select
        self.features = []
        self.score = {}

    def fit(self, X, y):
        return self._fit(X, y)

    def map_local_index(self, local_index, global_index):
        c = [global_index[i] for i in local_index]
        return c

    def calculate_mutual_information(self, x, y):
        H_y = drv.entropy(y)
        H_y_k = drv.entropy_conditional(y, x)
        return H_y - H_y_k

    def calculate_new_feature_redundancy_term(self, Xk, Xj, y):
        H_k = drv.entropy(Xk)
        H_k_j = drv.entropy_conditional(Xk, Xj)
        mi_k_j = H_k - H_k_j

        H_y_j = drv.entropy_conditional(y, Xj)
        H_k_y_j = drv.entropy_conditional(Xk, y, Xj)
        cmi = H_k_j + H_y_j - H_k_y_j

        return (mi_k_j, cmi)

    def _fit(self, X, y):
        # stage 1 - initialization
        S = list()
        n_features = X.shape[1]
        features_index = [i for i in range(n_features)]

        # stage 2 - calculate the mutual information between the class and each candidate feature
        mi_features_target = dict()
        for i in range(n_features):
            mi_features_target[i] = self.calculate_mutual_information(X[:, i], y)
        mi_features_target = dict(zip(features_index, mutual_info_classif(X, y)))

        # stage 3 - select the first feature
        S.append(max(mi_features_target, key=mi_features_target.get))

        # stage 4 - greedy selection
        while len(S < self.n_features_to_select):
            F = list(set(features_index) - set(S))
            feature_redundancy = dict()
            for X_k in F:
                for X_j in S:
                    feature_redundancy[X_k] = self.calculate_new_feature_redundancy_term(X[:, X_k], X[:, X_j], y)
                    mrmd =