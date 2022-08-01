import pandas as pd
import os
import numpy as np

path = '/sise/home/efrco/ML2/output/'

def concat_all_db():
    # path = '/sise/home/efrco/ML2/output/'
    for i, filename in enumerate(os.listdir(path)):
        if i == 0:
            df = pd.read_csv(path+filename)
            continue
        df_temp = pd.read_csv(path+filename)
        df = pd.concat([df, df_temp], ignore_index=True)
    df.to_csv(path+'All DB - All Results.csv')
    df = df.loc[df['Measure Type'] == 'AUC']
    df.to_csv(path + 'All DB - AUC.csv')


def create_new_results_df():
    df_result = pd.DataFrame(columns=['Dataset Name', 'Number of samples', 'Original Number of features',
                                  'Filtering Algorithm', 'Learning algorithm', 'Number of features selected (K)',
                                  'CV Method', 'Fold', 'Measure Type', 'Measure Value',
                                  'List of Selected Features Names', 'Selected Features scores'], dtype=object)
    return df_result


def print_best(df, db_name):
    auc_df = df.loc[df['Measure Type'] == 'AUC']
    auc_df = auc_df.loc[auc_df['Measure Value'].idxmax()]
    print(f"{db_name} --> Best Configuration ----- \n\t\tFiltering Algorithm: {auc_df['Filtering Algorithm']}"
          f"\n\t\tK: {auc_df['Number of features selected (K)']}"
          f"\n\t\tLearning algorithm: {auc_df['Learning algorithm']}"
          f"\n\t\tAUC Score: {auc_df['Measure Value']}")

def save_database(X, y, db_name, X_cols, X_idx):
    # path = '/sise/home/efrco/ML2/data_process/'
    # save database
    columns = list(X_cols)
    columns.extend(["Class"])
    y_reshape = y.reshape((-1, 1))
    an_array = np.append(X, y_reshape, axis=1)
    df = pd.DataFrame(data=an_array, columns=columns, index=X_idx, dtype=object)
    df.to_csv(path + db_name + ".csv")

def save_result(df, db_name):
    print_best(df, db_name)
    df.to_csv(path+f'{db_name}.csv')


def get_feature_names_by_idx(col_names, idx):
    return col_names[idx]


def write_result(df, db_name, sample_num, org_features, reduce_method, k, reduce_time, features_idx, score_features, models_scores):
    features_num = len(org_features)
    cv_method = models_scores[list(models_scores)[0]]['cv']
    folds = models_scores[list(models_scores)[0]]['folds']
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
