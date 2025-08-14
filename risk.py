#risk.py
import numpy as np
import pandas as pd

def calculate_atr(df, window=14):
    df['high'] = df['close'] * 1.02
    df['low'] = df['close'] * 0.98
    df['prev_close'] = df['close'].shift(1)

    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['prev_close']),
                               abs(df['low'] - df['prev_close'])))
    atr = pd.Series(tr).rolling(window=window).mean().iloc[-1]
    return atr

def calculate_stop_loss(current_price, atr, multiplier=1.5):
    return current_price - atr * multiplier

def calculate_take_profit(current_price, atr, multiplier=2.0):
    return current_price + atr * multiplier