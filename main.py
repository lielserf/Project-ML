from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.metrics import accuracy_score, make_scorer, matthews_corrcoef, roc_auc_score, precision_recall_curve
import read_data as read
from preprocessing import *
from sklearn.pipeline import Pipeline
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
import mifs
from sklearn.feature_selection import SelectFdr, f_classif, RFE, VarianceThreshold
from sklearn.svm import SVR
import feature_selection as FS
from ReliefF import ReliefF
from sklearn.model_selection import GridSearchCV, train_test_split
import utils
import classifier_switcher as models


""" --- Variables ---- """
cachedir = mkdtemp()
svm = SVR()
list_of_databases = ['CNS', 'Lymphoma', 'MLL', 'Ovarian', 'SRBCT', 'ayeastCC', 'bladderbatch', 'CLL',
                     'DLBCL', 'leukemiasEset', 'GDS4824', 'khan_train', 'NCI60_Affy',
                     'Nutt-2003-v2_BrainCancer.xlsx - Sayfa1', 'Risinger_Endometrial Cancer.xlsx - Sayfa1',
                     'madelon', 'ORL', 'RELATHE', 'USPS', 'Yale']


""" 
-------------------------------------
            Load Data 
------------------------------------- 
"""
X, y = read.read_data('bladderbatch')
y = y_to_categorical(y)


""" 
-------------------------------------
            Cross-Validation 
------------------------------------- 
"""
cv = utils.cv_djustment(X)


""" 
-------------------------------------
            Pipelines 
------------------------------------- 
"""
# preprocessing pipeline:
pre_estimators = [('convert', Convert()), ('Fill_Nan', FillNan()),
                  ('variance_threshold', VarianceThreshold()), ('norm', MinMaxScaler())]


# reduction pipeline
reduce_estimators = [('reduce_dim', 'passthrough')]
#N_FEATURES_OPTIONS = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]
N_FEATURES_OPTIONS = [1, 2]
# param_grid = [
#     {
#         "reduce_dim": [mifs.MutualInformationFeatureSelector('MRMR')],
#         "reduce_dim__k": N_FEATURES_OPTIONS,
#     },
#     {
#         "reduce_dim": [RFE(svm)],
#         "reduce_dim__n_features_to_select": N_FEATURES_OPTIONS,
#     },
# {
#         "reduce_dim": [ReliefF(n_features_to_keep=2)],
#         "reduce_dim__n_features_to_keep": N_FEATURES_OPTIONS,
#     },
# {
#         "reduce_dim": [SelectFdr(f_classif, alpha=0.1)],
#     },
# ]
param_grid = [
    {
        "reduce_dim": [mifs.MutualInformationFeatureSelector('MRMR')],
        "reduce_dim__k": N_FEATURES_OPTIONS,
    },
    {
        "reduce_dim": [RFE(svm)],
        "reduce_dim__n_features_to_select": N_FEATURES_OPTIONS,
    },
]
# models pipelines
models_estimators = [('clf', models.ClfSwitcher()),]
# parameters = [
#     {
#         'clf__estimator': [SVR()],
#     },
#     {
#         'clf__estimator': [KNeighborsClassifier()],
#     },
#     {
#         'clf__estimator': [RandomForestClassifier()],
#     },
#     {
#         'clf__estimator': [LogisticRegression()],
#     },
#     {
#         'clf__estimator': [GaussianNB()],
#     },
# ]
parameters = [

    {
        'clf__estimator': [KNeighborsClassifier()],
    },
    {
        'clf__estimator': [RandomForestClassifier()],
    },
    {
        'clf__estimator': [LogisticRegression()],
    },

]
acc = make_scorer(accuracy_score)
mcc = make_scorer(matthews_corrcoef)
auc = make_scorer(roc_auc_score)
pr_auc = make_scorer(precision_recall_curve)
#scoring_dict = {'ACC': acc, 'MCC': mcc, 'AUC': auc, 'PR-AUC': pr_auc}
scoring_dict = {'ACC': acc}

# main pipeline
pipe = Pipeline(pre_estimators + reduce_estimators + models_estimators, memory=cachedir)  # load the another estimators
grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid, cv=cv, scoring=scoring_dict, refit=False)
grid.fit(X, y)
print(grid.cv_results_)

rmtree(cachedir)