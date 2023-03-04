import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error

from .preprocess import (categorical_columns_encoding,
                         continuous_feature_scaling, merging_the_columns,
                         spliting_train_and_test_data)


def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def build_model(data: pd.DataFrame) -> dict[str, str]:
    X = data[['GarageCars','HouseStyle','PoolArea', 'GrLivArea']]
    y = data[['SalePrice']]
    X_train, X_test, y_train, y_test = split_train_test_data(X, y, split_ratio=0.25)
    scaled_columns_train = continuous_features_scaling(X_train[['PoolArea', 'GrLivArea']])
    encoded_columns_train = categorical_columns_encoding(X_train[['HouseStyle', 'GarageCars']])
    X_train_update = merging_the_columns(scaled_columns_train, encoded_columns_train)
    reg_multiple = LinearRegression()
    reg_multiple_update = reg_multiple.fit(X_train_update, y_train)
    joblib.dump(reg_multiple_update, "../models/model.joblib")
    encoder = joblib.load("../models/encoder.joblib")
    scaler = joblib.load("../models/scaler.joblib")
    scaled_columns_test = scaler.transform(X_test[['PoolArea', 'GrLivArea']])
    encoded_columns_test = encoder.transform(X_test[['HouseStyle', 'GarageCars']])
    X_test_update = merging_the_columns(scaled_columns_test, encoded_columns_test)
    y_pred = reg_multiple_update.predict(X_test_new)
    rmsle_score = compute_rmsle(y_test, y_pred)
    return{"rmsle": rmsle_score}