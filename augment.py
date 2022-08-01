from main import *
from sklearn.decomposition import KernelPCA
from imblearn.over_sampling import SMOTE
from fs.feature_selection import *
from utils.write_results import *

def add_pca(X_train, X_test):
    pca_linear = KernelPCA(kernel='linear')
    pca_rbf = KernelPCA(kernel='rbf')

    # fit transform
    train_pca_linear_features = pca_linear.fit_transform(X_train)
    train_pca_rbf_features = pca_rbf.fit_transform(X_train)

    # add to train
    X_train = np.append(X_train, train_pca_linear_features, 1)
    X_train = np.append(X_train, train_pca_rbf_features, 1)

    # add to test
    test_pca_linear_features = pca_linear.transform(X_test)
    test_pca_rbf_features = pca_rbf.transform(X_test)
    X_test = np.append(X_test, test_pca_linear_features, 1)
    X_test = np.append(X_test, test_pca_rbf_features, 1)
    return X_train, X_test


def add_smote(X_train, y_train):
    sm = SMOTE(random_state=42)
    return sm.fit_resample(X_train, y_train)


def augment_db(df, db, fs_method, k, clf):
    clf = get_models()[clf]
    X, y, db_name, X_cols, X_idx, multi_class = load_data(db)
    X, y = per_processing(X, y)

    retucer_dict = run_reducer(fs_method, X, y)
    X_new = X[:, :][:, retucer_dict['features'][:k]]
    cv_name, cv = cv_djustment(len(X))

    if multi_class:
        y = pd.get_dummies(y).to_numpy()
        clf = OneVsRestClassifier(clf)

    fit_time, pred_time, split = 0, 0, 1
    pred, pred_prob, y_test_all = [], [], []
    for train, test in cv.split(X_new, y):
        X_train, X_test = X_new[train], X_new[test]
        y_train, y_test = y[train], y[test]

        # augment
        X_train, X_test = add_pca(X_train, X_test)
        X_train, y_train = add_smote(X_train, y_train)

        y_test_all.extend(y_test.tolist())
        clf = clone(clf)

        # fit the model
        if split == 1:
            start = time.time()
        clf.fit(X_train, y_train)
        if split == 1:
            fit_time = time.time() - start
            start = time.time()

        # predict
        y_pred = clf.predict(X_test)
        if split == 1:
            pred_time = time.time() - start
        pred.extend(y_pred.tolist())

        # score
        if multi_class:
            if hasattr(clf, "decision_function"):
                y_score = clf.decision_function(X_test)
            else:
                y_score = clf.predict_proba(X_test)
            pred_prob.extend(y_score.tolist())
        else:
            pred_prob.extend((clf.predict_proba(X_test)).tolist())
        split += 1

        # results conclusion
    scoring_dict = get_scores(y_test_all, pred, pred_prob, multi_class)
    scoring_dict['fit_time'] = fit_time
    scoring_dict['pred_time'] = pred_time
    scoring_dict['cv'] = cv_name
    scoring_dict['folds'] = split

    all_score = {clf: scoring_dict}
    df = write_result(df, db_name, len(X_idx), X_cols, fs_method+"_Aug", k, retucer_dict['time'], retucer_dict['features'][:k],
                          retucer_dict['scorer'][:k], all_score)
    return df


