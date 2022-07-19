from sklearn.model_selection import LeavePOut, LeaveOneOut, KFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, precision_recall_curve, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import time
import pandas as pd
import numpy as np
from sklearn.base import clone


def get_models():
    return {'SVM': SVC(),
            'KNN': KNeighborsClassifier(),
            'RandomForest': RandomForestClassifier(),
            'LogisticsRegression': LogisticRegression(),
            'NB': GaussianNB()}


def get_score_multi(y_test, y_score, score):
    score['ACC'] = accuracy_score(y_test, y_score)
    score['MCC'] = matthews_corrcoef(y_test, y_score)
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
    score['AUC'] = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    score['PR-AUC'] = auc(precision, recall)
    return score


def get_score_binary(y_test, y_pred, y_score, score):
    score['ACC'] = accuracy_score(y_test, y_pred)
    score['MCC'] = matthews_corrcoef(y_test, y_pred)
    score['AUC'] = roc_auc_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    score['PR-AUC'] = auc(precision, recall)
    return score


def cv_djustment(df):
    sample_size = len(df)
    if sample_size < 50:
        return 'Leave-pair-out', LeavePOut(2)
    elif 50 <= sample_size < 100:
        return 'LOOCV', LeaveOneOut()
    elif 100 <= sample_size < 1000:
        return '10-Fold-CV', KFold(n_splits=10, random_state=100)
    else:
        return '5-Fold-CV', KFold(n_splits=5, random_state=100)


def evaluate_model(model, X, y, multi_class=False):
    fit_time, pred_time = 0, 0
    cv_name, cv = cv_djustment(X)
    scoring_dict = {'ACC': [], 'MCC': [], 'AUC': [],
                    'PR-AUC': []}
    if multi_class:
        y = pd.get_dummies(y)
    for s, train, test in enumerate(cv.split(X, y)):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        model = clone(model)

        # fit the model
        if s == 0:
            start = time.time()
        model.fit(X_train, y_train)
        if s == 0:
            fit_time = time.time() - start
            start = time.time()

        # predict
        y_pred = model.predict(X_test)
        if s == 0:
            pred_time = time.time() - start

        # score
        if multi_class:
            model_score = get_score_multi(y_test, model.decision_function(X_test), scoring_dict)
        else:
            model_score = get_score_binary(y_test, y_pred, model.decision_function(X_test), scoring_dict)
        for metric in scoring_dict:
            scoring_dict[metric].append(model_score[metric])

    # results conclusion
    scoring_dict['fit_time'] = fit_time
    scoring_dict['pred_time'] = pred_time
    scoring_dict['cv'] = cv_name
    scoring_dict['folds'] = s+1
    for metric in scoring_dict:
        scoring_dict[metric] = np.mean(scoring_dict[metric])

    return scoring_dict