from sklearnex import patch_sklearn
patch_sklearn()
import read_data as read
from preprocessing import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from fs.feature_selection import run_reducer
from utils.write_results import *
from utils.clf_models import *
import time


def load_data(db):
    """
    Read and load the database into dataframe
    :param db: name or index of the db
    :return: X, y and metadata
    """
    db_name, X, y = read.read_data(1)
    multi_class = y.nunique() > 2
    y = y_to_categorical(y)
    X_cols = X.columns
    X_idx = X.index
    return X, y, db_name, X_cols, X_idx, multi_class


def per_processing(X, y):
    """
    Run pipeline pf preprocessing methods
    """
    pre_estimators = [('convert', Convert()), ('Fill_Nan', FillNan()), ('variance_threshold', VarianceThreshold()),
                      ('scaler', StandardScaler(with_std=False)), ('norm', PowerTransformer())]
    pipe = Pipeline(pre_estimators)
    pipe.fit(X, y)
    X = pipe.transform(X)
    return X, y


def get_reducers(X, y):
    """
    Reduce the feature set to 100
    :param X:
    :param y:
    :return: dictionary contains the top-100 features and their scores
    """

    reducers = {'mRmr': {},
                'RFE': {},
                'ReliefF': {},
                'Fdr': {}
                }
    for r in reducers:
        reducers[r] = run_reducer(r, X, y)
    return reducers


def main(db):
    """
    Main Function - run all the program
    :return: Write to disk csv file with results
    """
    X, y, db_name, X_cols, X_idx, multi_class = load_data(db)
    X, y = per_processing(X, y)
    reducers = get_reducers(X, y)
    cv_name, cv = cv_djustment(len(X))
    N_FEATURES_OPTIONS = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]

    df_res = create_new_results_df()
    for r in reducers:
        for k in reversed(N_FEATURES_OPTIONS):
            X_new = X[:, :][:, reducers[r]['features'][:k]]
            models = get_models()
            models_scores = {}
            for clf, model in models.items():
                models_scores[clf] = evaluate_model(model, X_new, y, cv_name, cv, multi_class)
            df_res = write_result(df_res, db_name, len(X_idx), X_cols, r, k, reducers[r]['time'], reducers[r]['features'][:k], reducers[r]['scorer'][:k], models_scores)
    save_result(df_res, db_name)


if __name__ == "__main__":
    args = 1
    if args == 'all':
        for i in range(1,21):
            main(1)
    else:
        main(args)
