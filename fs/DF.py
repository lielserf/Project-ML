from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif as MIC
import numpy as np


class ReduceDF():

    def __init__(
        self,
        n_features_to_select,

    ):
        self.n_features_to_select = n_features_to_select
        self.features = []
        self.score = []


    def fit(self, X, y):
        """Fit the RFE model and then the underlying estimator on the selected features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        return self._fit(X, y)
    def get_information_gain_features(X, y):
        mi_score = MIC(X,y)
        mi_score_selected_index = np.where(mi_score > 0.25)[0]

        return

    def _fit(self, X, y):
        # Parameter step_score controls the calculation of self.scores_
        # step_score is not exposed to users
        # and is used when implementing RFECV
        # self.scores_ will not be calculated when calling _fit through fit

        dict_group_df = {}
        dict_group_sub = {}

        n_features = X.shape[1]
        # number of features in each group
        k = int(X.shape[0] / 2)
        # number of group with k features
        n = int(n_features / k)
        index_column = [i for i in range(X.shape[1])]
        for i in range(n):
            d_i = X[:,index_column[i*k:i*k+k]]
            dict_group_df[i] = d_i

        # the best results is for svm with information gain of 25%
        for k in dict_group_sub.keys():
            dict_group_sub[k] = self.get_information_gain_features(X, y)
            # X_2 = X[:, mi_score_selected_index]




        # ranking_ = np.ones(n_features, dtype=int)
        #
        # if step_score:
        #     self.scores_ = []
        #
        # # Elimination
        # while np.sum(support_) > n_features_to_select:
        #     # Remaining features
        #     features = np.arange(n_features)[support_]
        #
        #     # Rank the remaining features
        #     estimator = clone(self.estimator)
        #     if self.verbose > 0:
        #         print("Fitting estimator with %d features." % np.sum(support_))
        #
        #     estimator.fit(X[:, features], y, **fit_params)

        #     # Get importance and rank them
        #     importances = _get_feature_importances(
        #         estimator,
        #         self.importance_getter,
        #         transform_func="square",
        #     )
        #     ranks = np.argsort(importances)
        #
        #     # for sparse case ranks is matrix
        #     ranks = np.ravel(ranks)
        #
        #     # Eliminate the worse features
        #     threshold = min(step, np.sum(support_) - n_features_to_select)
        #
        #     # Compute step score on the previous selection iteration
        #     # because 'estimator' must use features
        #     # that have not been eliminated yet
        #     if step_score:
        #         self.scores_.append(step_score(estimator, features))
        #     support_[features[ranks][:threshold]] = False
        #     ranking_[np.logical_not(support_)] += 1
        #
        # # Set final attributes
        # features = np.arange(n_features)[support_]
        # self.estimator_ = clone(self.estimator)
        # self.estimator_.fit(X[:, features], y, **fit_params)
        #
        # # Compute step score when only n_features_to_select features left
        # if step_score:
        #     self.scores_.append(step_score(self.estimator_, features))
        # self.n_features_ = support_.sum()
        # self.support_ = support_
        # self.ranking_ = ranking_
        #
        # return self


    def score(self, X, y, **fit_params):
        """Reduce X to the selected features and return the score of the underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.

        **fit_params : dict
            Parameters to pass to the `score` method of the underlying
            estimator.

            .. versionadded:: 1.0

        Returns
        -------
        score : float
            Score of the underlying base estimator computed with the selected
            features returned by `rfe.transform(X)` and `y`.
        """
        check_is_fitted(self)
        return self.estimator_.score(self.transform(X), y, **fit_params)


