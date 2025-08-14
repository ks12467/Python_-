# predictor.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

def prepare_data(df, window=30):
    data = df[['close']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window, len(scaled) - 1):
        X.append(scaled[i-window:i])
        y.append(scaled[i+1][0])

    return np.array(X), np.array(y), scaler

def predict_next(df, market):
    X, y, scaler = prepare_data(df)
    model_path = os.path.join("models", f"{market}.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 없음: {market}")

    model = load_model(model_path)
    pred_scaled = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))[0][0]
    pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
    return pred_price