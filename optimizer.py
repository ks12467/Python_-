import os
import time
import numpy as np
from data_loader import get_ohlcv, get_all_krw_markets
from model import build_model
from sklearn.preprocessing import MinMaxScaler

MODEL_DIR = "models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def prepare_data(df, window=30):
    data = df[['close']].values
    if len(data) <= window + 1:
        raise ValueError("📉 데이터 부족")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(window, len(scaled) - 1):
        X.append(scaled[i-window:i])
        y.append(scaled[i+1][0])
    return np.array(X), np.array(y), scaler

def train_and_save_model(market):
    try:
        df = get_ohlcv(market)
        if df is None or df.empty or len(df) < 50:
            raise ValueError("❌ 데이터 부족")
        X, y, scaler = prepare_data(df)
        model = build_model((X.shape[1], X.shape[2]))
        model.fit(X, y, epochs=50, verbose=0)
        model.save(os.path.join(MODEL_DIR, f"{market}.h5"))
        print(f"✅ {market} 모델 저장 완료")
    except Exception as e:
        print(f"❌ {market} 실패: {e}")

def main():
    markets = get_all_krw_markets()
    print(f"총 {len(markets)}개 코인 학습 시작")

    for i, market in enumerate(markets, 1):
        print(f"[{i}/{len(markets)}] {market} 학습 중...")
        train_and_save_model(market)
        time.sleep(0.3)  # API Rate Limit 방지

    print("✅ 전체 최적화 학습 완료")

if __name__ == "__main__":
    main()