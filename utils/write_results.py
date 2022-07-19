import pandas as pd

df_result = pd.DataFrame(columns=['Dataset Name', 'Number of samples', 'Original Number of features',
                                  'Filtering Algorithm', 'Learning algorithm', 'Number of features selected (K)',
                                  'CV Method', 'Fold', 'Measure Type', 'Measure Value',
                                  'List of Selected Features Names', 'Selected Features scores'])


def get_feature_names_by_idx(col_names, idx):
    pass


def write_result(db_name, sample_num, org_features, reduce_method, clf_name, k, reduce_time, features_idx, score_features, models_scores):
    features_num = len(org_features)
    cv_method = models_scores['cv']
    folds = models_scores['folds']
    selected_features = get_feature_names_by_idx(org_features, features_idx)
    for metric in models_scores:
        if metric not in ['cv', 'folds']:
            df_result.append([db_name, sample_num, features_num, reduce_method, clf_name, k, cv_method, folds, metric,
                              models_scores[metric], selected_features, score_features])
    if k == 100:
        df_result.append([db_name, sample_num, features_num, reduce_method, clf_name, k, cv_method, folds, 'reduce_time',
                        reduce_time, selected_features, score_features])
