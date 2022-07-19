from sklearnex import patch_sklearn
patch_sklearn()
import read_data as read
from preprocessing import *
from sklearn.pipeline import Pipeline
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.feature_selection import SelectFdr, f_classif, RFE, VarianceThreshold
from sklearn.svm import SVC
from fs.feature_selection import get_features_and_scorer
from fs.MRMR import mrmr
from fs.reliefF import reliefF, feature_ranking
from ReliefF import ReliefF
from utils.write_results import *
from utils.clf_models import *
import time

""" 
-------------------------------------
            Variables 
------------------------------------- 
"""
cachedir = mkdtemp()
svm = SVC(kernel="linear")
list_of_databases = ['CNS', 'Lymphoma', 'MLL', 'Ovarian', 'SRBCT', 'ayeastCC', 'bladderbatch', 'CLL',
                     'DLBCL', 'leukemiasEset', 'GDS4824', 'khan_train', 'NCI60_Affy',
                     'Nutt-2003-v2_BrainCancer.xlsx - Sayfa1', 'Risinger_Endometrial Cancer.xlsx - Sayfa1',
                     'madelon', 'ORL', 'RELATHE', 'USPS', 'Yale']

""" 
-------------------------------------
            Load Data 
------------------------------------- 
"""
db_name, X, y = read.read_data('khan_train')
multi_class = y.nunique() > 2
y = y_to_categorical(y)
X_cols = X.columns
X_idx = X.index

""" 
----------------------------------------------------- 
            PreProcessing - Pipelines 
----------------------------------------------------- 
"""
# preprocessing pipeline:
pre_estimators = [('convert', Convert()), ('Fill_Nan', FillNan()), ('variance_threshold', VarianceThreshold()),
                  ('scaler', StandardScaler(with_std=False)), ('norm', PowerTransformer())]
pipe = Pipeline(pre_estimators, memory=cachedir)
pipe.fit(X, y)
X = pipe.transform(X)

""" 
----------------------------------------------------- 
                  Reducing 
----------------------------------------------------- 
"""
# mRmr
start = time.time()
idx, J_CMI, MIfy = mrmr(X, y, n_selected_features=100)
mrmr_time = time.time() - start

# RFE
RFE = RFE(svm, n_features_to_select=1)

# ReliefF
start = time.time()
score = reliefF(X, y)
relieff_time = time.time() - start
relieff_score = np.sort(score)[:100:-1]
idx_relieff = feature_ranking(score)[:100]

# FDR
Fdr = SelectFdr(f_classif, alpha=0.1)

reducers = {'mRmr': {'scorer': J_CMI, 'features_idx': idx, 'time': mrmr_time},
            'RFE': {'func': RFE, 'scorer': [], 'features': [], 'time': 0},
            'ReliefF': {'func': ReliefF, 'scorer': relieff_score, 'features': idx_relieff, 'time': relieff_time},
            'Fdr': {'func': Fdr, 'scorer': [], 'features': [], 'time': 0}}

"""
-------------------------------------
    Main Loop
-------------------------------------
"""

N_FEATURES_OPTIONS = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]
for r in reducers:
    if r not in ['mRmr', 'ReliefF']:
        start = time.time()
        reducers[r]['func'].fit(X, y)
        timeit = time.time() - start
        reducers[r]['features'], reducers[r]['scorer'] = get_features_and_scorer(r, reducers[r]['func'])
        reducers[r]['time'] = timeit
    for k in N_FEATURES_OPTIONS:
        X_new = X[:, reducers[r]['features'][:k]]
        models = get_models()
        models_scores = {}
        for clf, model in models.items():
            models_scores[clf] = evaluate_model(model, X_new, y, multi_class)
        write_result(db_name, len(X_idx), X_cols, r, clf, k, reducers[r]['time'], reducers[r]['features'][:k], reducers[r]['scorer'][:k], models_scores)
rmtree(cachedir)
