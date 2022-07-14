import read_data as read
from preprocessing import *
from sklearn.pipeline import Pipeline
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold


""" --- Variables ---- """
cachedir = mkdtemp()
list_of_databases = ['CNS', 'Lymphoma', 'MLL', 'Ovarian', 'SRBCT', 'ayeastCC', 'bladderbatch', 'CLL',
                     'DLBCL', 'leukemiasEset', 'GDS4824', 'khan_train', 'NCI60_Affy',
                     'Nutt-2003-v2_BrainCancer.xlsx - Sayfa1', 'Risinger_Endometrial Cancer.xlsx - Sayfa1',
                     'madelon', 'ORL', 'RELATHE', 'USPS', 'Yale']


""" --- Load Data ---- """
X, y = read.read_data(1)
y = y_to_categorical(y)


""" --- Pipelines ---- """
# preprocessing pipeline:
pre_estimators = [('convert', Convert()), ('Fill_Nan', FillNan()), ('variance_threshold', VarianceThreshold()), ('norm', MinMaxScaler())]
pipe_pre = Pipeline(pre_estimators, memory=cachedir)

# reduction pipeline
our1 = None
our2 = None
mRMR =

pipe_pre.fit(X, y)
t = pipe_pre[:-1].get_feature_names_out()
print(t)