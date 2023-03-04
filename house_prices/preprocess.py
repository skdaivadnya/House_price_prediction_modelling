import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def spliting_train_and_test_data(X, y, ratio):
    return train_test_split(X, y, test_size= ratio, random_state=0)


def continuous_feature_scaling(name_of_columns):
    scaler = StandardScaler()
    scaler.fit(name_of_columns)
    joblib.dump(scaler, "../models/scaler.joblib")
    s = scaler.transform(name_of_columns)
    return(s)


def categorical_columns_encoding(name_of_column):
    encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    encoder.fit(def categorical_columns_encoding(name_of_column):
)
    joblib.dump(encoder, "../models/encoder.joblib")
    e = encoder.transform(def categorical_columns_encoding(name_of_column):
)
    return(e)


def merging_the_columns(s, e):
    col_mer = np.hstack((s, e))
    return(col_mer)