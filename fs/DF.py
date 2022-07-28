import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif
import random


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

    def svm(self,X,y):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=4)
        if len(np.unique(y)) > 2:
            clf = SVC()
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred) * 100

        else:
            clf = SVC()
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred) * 100
        return accuracy

    def get_information_gain_features(self,X, y):
        selector = SelectPercentile(mutual_info_classif, percentile=25)
        X_reduced = selector.fit_transform(X, y)
        cols = selector.get_support(indices=True)
        score_select_featuers = [selector.scores_[i] for i in cols]
        return cols, score_select_featuers

    def _fit(self, X, y):

        dict_group_sub = {}
        dict_group_original_index= {}

        n_features = X.shape[1]
        # number of features in each group
        k = int(X.shape[0] / 2)
        # number of group with k features
        n = int(n_features / k)
        index_column = [i for i in range(X.shape[1])]
        random.shuffle(index_column)

        for i in range(n):
            d_i = X[:, index_column[i*k:i*k+k]]
            dict_group_original_index[i] = index_column[i*k:i*k+k]
            dict_group_sub[i] = self.get_information_gain_features(d_i, y)

        global_index_select = self.map_local_index(dict_group_sub[0][0], dict_group_original_index[0])
        x_sub = X[:, global_index_select]
        basline = self.svm(x_sub, y)
        s_aux = global_index_select
        self.score = dict(zip(dict_group_sub[0][0], dict_group_sub[0][1]))

        for i in range(1,n):
            s_i = dict_group_sub[i]
            global_index_select = self.map_local_index(s_i[0], dict_group_original_index[i])
            s_aux.extend(global_index_select)
            x_sub = X[:, s_aux]
            accuracy = self.svm(x_sub, y)

            if accuracy > basline:
                basline = accuracy
                for i in range(len(global_index_select)):
                    self.score[global_index_select[i]] = s_i[1][i]

            else:
                s_aux = [x for x in s_aux if x not in global_index_select]

        sorted_score = sorted(self.score.items(), key=lambda x: x[1])
        sorted_dict = {k: v for k, v in sorted_score}

        self.features = list(sorted_dict.keys())
        self.score = list(sorted_dict.values())
