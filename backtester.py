#backtester
import pandas as pd
from analyzer import get_rsi, get_macd_signal, is_uptrend
from predictor import predict_next
import os

def backtest_strategy(df, market, threshold=0.03, rsi_range=(45, 65)):
    history = []
    for i in range(60, len(df)-1):
        sample = df.iloc[:i].copy()
        true_price = df['close'].iloc[i+1]
        current_price = df['close'].iloc[i]

        try:
            rsi = get_rsi(sample)
            macd, signal = get_macd_signal(sample)

            if not (rsi and macd and signal):
                continue

            if rsi_range[0] < rsi < rsi_range[1] and macd > signal and is_uptrend(sample):
                predicted = predict_next(sample, market)
                change = (true_price - current_price) / current_price
                history.append({
                    'index': i,
                    'predicted': predicted,
                    'actual': true_price,
                    'current': current_price,
                    'return': round(change * 100, 2),
                    'success': change >= threshold
                })
        except:
            continue

    df_result = pd.DataFrame(history)
    os.makedirs("logs/backtests", exist_ok=True)
    df_result.to_csv(f"logs/backtests/{market}_backtest.csv", index=False)
    return df_result