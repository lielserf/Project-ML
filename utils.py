from sklearn.model_selection import LeavePOut, LeaveOneOut, KFold


def cv_djustment(df):
    sample_size = len(df)
    if sample_size < 50:
        return LeavePOut(2)
    elif 50 <= sample_size < 100:
        return LeaveOneOut()
    elif 100 <= sample_size < 1000:
        KFold(n_splits=10, random_state=100)
    else:
        KFold(n_splits=5, random_state=100)