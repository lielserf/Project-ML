import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif
from skfeature.function.statistical_based import CFS
# By having an ordered ranking, features with similar relevance to the class will be in the
# same subset, which will facilitate the task of the subset filter which will be applied later
from fs.reliefF import reliefF, feature_ranking
from xgboost import XGBClassifier
#
# class ReduceDRF0Improve():
#
#     def __init__(
#         self,
#         n_features_to_select,
#
#     ):
#         self.n_features_to_select = n_features_to_select
#         self.features = []
#         self.score = {}
#
#     def fit(self, X, y):
#         return self._fit(X, y)
#
#     def set_score(self, number_featuers):
#         self.score=[]
#         for i in range(number_featuers):
#             if i in self.features:
#                 self.score[i] = 1
#             else:
#                 self.score[i] = 0
#
#
#     def map_local_index(self, local_index, global_index):
#         c = [global_index[i] for i in local_index]
#         return c
#
#     # ranking the original features before generating the subsets
#     # the method we choose is informatoin gain
#     def partition_dataset(self,X, y):
#         selector = SelectKBest(mutual_info_classif, k=X.shape[1])
#         X_reduced = selector.fit_transform(X, y)
#         cols = selector.get_support(indices=True)
#
#         ranking_dict = dict(zip(cols, selector.scores_))
#         sorted_score = sorted(ranking_dict.items(), key=lambda x: x[1], reverse=True)
#         sorted_dict = {k: v for k, v in sorted_score}
#
#         return list(sorted_dict.keys())
#
#     def CFS(self,X, y):
#         idx = CFS.cfs(X, y)
#         idx_list = np.ndarray.tolist(idx)
#         if len(idx_list) > X.shape[1]:
#             idx_list.remove(X.shape[1])
#         return idx_list
#
#     def svm(self,X,y):
#         x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
#                                                             random_state=4)
#         clf = SVC()
#         clf.fit(x_train, y_train)
#         y_pred = clf.predict(x_test)
#         accuracy = accuracy_score(y_test, y_pred) * 100
#         return accuracy
#
#     def get_information_gain_features(self,X, y):
#         selector = SelectPercentile(mutual_info_classif, percentile=25)
#         X_reduced = selector.fit_transform(X, y)
#         cols = selector.get_support(indices=True)
#         return cols
#
#
#     def _fit(self, X, y):
#         dict_group_sub = {}
#         dict_group_original_index= {}
#
#         n_features = X.shape[1]
#         # number of features in each group
#         k = int(X.shape[0] / 2)
#         # number of group with k features
#         n = int(n_features / k)
#         index_column = self.partition_dataset(X,y)
#
#         for i in range(n + 1):
#             if i == n:
#                 d_i = X[:, index_column[i * k:]]
#                 dict_group_original_index[i] = index_column[i * k:]
#
#             else:
#                 d_i = X[:, index_column[i * k:i * k + k]]
#                 dict_group_original_index[i] = index_column[i * k:i * k + k]
#             dict_group_sub[i] = self.CFS(d_i, y)
#
#         global_index_select = self.map_local_index(dict_group_sub[0], dict_group_original_index[0])
#         x_sub = X[:, global_index_select]
#         baseline = self.svm(x_sub, y)
#         S = global_index_select
#
#         for i in range(1, n+1):
#             s_i = dict_group_sub[i]
#             global_index_select = self.map_local_index(s_i, dict_group_original_index[i])
#             S.extend(global_index_select)
#             x_sub = X[:, S]
#             accuracy = self.svm(x_sub, y)
#
#             if accuracy > baseline:
#                 baseline = accuracy
#
#                 # Choose the best subset
#                 # print("Before Choose:", len(S))
#                 s_aux_local = self.get_information_gain_features(x_sub, y)
#                 s_aux_global = self.map_local_index(s_aux_local, S)
#                 x_sub = X[:, s_aux_global]
#                 accuracy = self.svm(x_sub, y)
#                 if accuracy > baseline:
#                     baseline = accuracy
#                     S = s_aux_global
#
#             else:
#                 S = [x for x in S if x not in global_index_select]
#
#         self.features = np.array(S)
#         self.score = np.array(self.set_score(X.shape[1]))


class ReduceDRF0Improve():

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

    # ranking the original features before generating the subsets
    # the method we choose is informatoin gain
    def partition_dataset(self, X, y):
        selector = SelectKBest(mutual_info_classif, k=X.shape[1])
        X_reduced = selector.fit_transform(X, y)
        cols = selector.get_support(indices=True)

        ranking_dict = dict(zip(cols, selector.scores_))
        sorted_score = sorted(ranking_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_dict = {k: v for k, v in sorted_score}

        return list(sorted_dict.keys())

    def CFS(self, X, y):
        idx = CFS.cfs(X, y)
        idx_list = np.ndarray.tolist(idx)
        score_select_featuers = [1 for i in idx_list]
        if len(idx_list) > X.shape[1]:
            idx_list.remove(X.shape[1])
        return idx_list, score_select_featuers

    def svm(self, X, y):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=4)
        clf = SVC()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        return accuracy

    def get_information_gain_features(self, X, y):
        selector = SelectPercentile(mutual_info_classif, percentile=25)
        X_reduced = selector.fit_transform(X, y)
        cols = selector.get_support(indices=True)
        score_select_featuers = [selector.scores_[i] for i in cols]
        return cols, score_select_featuers



    def _fit(self, X, y):
        dict_group_sub = {}
        dict_group_original_index = {}

        n_features = X.shape[1]
        # number of features in each group
        k = int(X.shape[0] / 2)
        # number of group with k features
        n = int(n_features / k)
        index_column = self.partition_dataset(X, y)

        for i in range(n + 1):
            if i == n:
                d_i = X[:, index_column[i * k:]]
                dict_group_original_index[i] = index_column[i * k:]

            else:
                d_i = X[:, index_column[i * k:i * k + k]]
                dict_group_original_index[i] = index_column[i * k:i * k + k]
            dict_group_sub[i] = self.CFS(d_i, y)

        global_index_select = self.map_local_index(dict_group_sub[0][0], dict_group_original_index[0])
        x_sub = X[:, global_index_select]
        basline = self.svm(x_sub, y)
        S = global_index_select
        self.score = dict(zip(dict_group_sub[0][0], dict_group_sub[0][1]))

        for i in range(1, n + 1):
            s_i = dict_group_sub[i]
            global_index_select = self.map_local_index(s_i[0], dict_group_original_index[i])
            S.extend(global_index_select)
            x_sub = X[:, S]
            accuracy = self.svm(x_sub, y)

            if accuracy > basline:
                basline = accuracy
                for k in range(len(global_index_select)):
                    self.score[global_index_select[k]] = s_i[1][k]
                # Choose the best subset
                s_aux_global, scores = self.get_information_gain_features(x_sub, y)
                s_aux_temp = self.map_local_index(s_aux_global, s_aux)
                x_sub = X[:, s_aux_temp]
                accuracy = self.svm(x_sub, y)
                if accuracy > basline:
                    basline = accuracy
                    s_aux = s_aux_temp
                    for k in range(len(s_aux)):
                        self.score[s_aux[k]] = scores[k]

            else:
                S = [x for x in S if x not in global_index_select]

        sorted_score = sorted(self.score.items(), key=lambda x: x[1])
        sorted_dict = {k: v for k, v in sorted_score}

        self.features = np.array(list(sorted_dict.keys()))
        self.score = np.array(list(sorted_dict.values()))
