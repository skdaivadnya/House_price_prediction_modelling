import numpy as np
import pandas as pd
import joblib
from .preprocess import merge_columns


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    scaler = joblib.load("../models/scaler.joblib")
    encoder = joblib.load("../models/encoder.joblib")
    model = joblib.load("../models/model.joblib")
   

    encoded_test_columns = encoder.transform(
        input_data[['HouseStyle', 'GarageCars']])
    scaled_test_columns = scaler.transform(
        input_data[['GrLivArea', 'PoolArea']])    

        
    processed_data = merge_columns(scaled_test_columns, encoded_test_columns)
    predictions = model.predict(processed_data)
    return(predictions)