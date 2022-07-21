import pandas as pd

def create_new_results_df():
    df_result = pd.DataFrame(columns=['Dataset Name', 'Number of samples', 'Original Number of features',
                                  'Filtering Algorithm', 'Learning algorithm', 'Number of features selected (K)',
                                  'CV Method', 'Fold', 'Measure Type', 'Measure Value',
                                  'List of Selected Features Names', 'Selected Features scores'])
    return df_result


def save_result(df, db_name):
    df.to_csv(f'{db_name}.csv')


def get_feature_names_by_idx(col_names, idx):
    return col_names[idx]


def write_result(df, db_name, sample_num, org_features, reduce_method, k, reduce_time, features_idx, score_features, models_scores):
    features_num = len(org_features)
    cv_method = models_scores['SVM']['cv']
    folds = models_scores['SVM']['folds']
    selected_features = get_feature_names_by_idx(org_features, features_idx).tolist()
    selected_features = ';'.join(selected_features)
    score_features = [str(i) for i in score_features]
    score_features = ';'.join(score_features)
    if k == 100:
        df.loc[len(df)] = [db_name, sample_num, features_num, reduce_method, "---", k, cv_method, folds, 'reduce_time',
                        reduce_time, selected_features, score_features]
    for clf in models_scores:
        for metric in models_scores[clf]:
            if metric not in ['cv', 'folds']:
                df.loc[len(df)] = [db_name, sample_num, features_num, reduce_method, clf, k, cv_method, folds, metric,
                                  models_scores[clf][metric], selected_features, score_features]
    return df




