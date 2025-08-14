# train.py

import os
import time
import numpy as np
from data_loader import get_ohlcv, get_all_krw_markets
from model import build_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import save_model

# 🔧 디렉토리 설정
MODEL_DIR = "models"
LOG_DIR = "logs"
FAILED_LOG_PATH = os.path.join(LOG_DIR, "failed_markets.log")
RETRY_FAILED_ONLY = False  # True로 설정 시 실패한 코인만 재시도

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 📦 학습 데이터 준비 함수
def prepare_data(df, window=30):
    data = df[['close']].values
    if len(data) <= window + 1:
        raise ValueError("📉 데이터가 너무 적음")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window, len(scaled) - 1):
        X.append(scaled[i-window:i])
        y.append(scaled[i+1][0])

    return np.array(X), np.array(y), scaler

# 🧠 모델 학습 및 저장
def train_and_save_model(market):
    try:
        df = get_ohlcv(market, count=200, unit=5)
        if df is None or df.empty or len(df) < 50:
            raise ValueError("📉 학습 데이터 부족")

        X, y, scaler = prepare_data(df)
        model = build_model((X.shape[1], X.shape[2]))
        model.fit(X, y, epochs=30, verbose=0)

        model_path = os.path.join(MODEL_DIR, f"{market}.h5")
        model.save(model_path)
        print(f"✅ {market} 모델 저장 완료")

    except Exception as e:
        print(f"❌ {market} 학습 실패: {e}")
        with open(FAILED_LOG_PATH, "a") as f:
            f.write(f"{market} 실패: {e}\n")

# ♻️ 실패한 코인 재시도
def get_failed_markets():
    if not os.path.exists(FAILED_LOG_PATH):
        return []
    with open(FAILED_LOG_PATH, 'r') as f:
        lines = f.readlines()
    return list(set([line.split()[0] for line in lines]))

def main():
    # 🔄 실패한 코인만 재학습할지 여부
    if RETRY_FAILED_ONLY:
        print("🔁 실패한 코인만 재시도 중...\n")
        markets = get_failed_markets()
    else:
        print("🧠 전체 코인 학습 중...\n")
        markets = get_all_krw_markets()

    total = len(markets)
    print(f"📦 총 {total}개 코인 학습 예정\n")

    for i, market in enumerate(markets, 1):
        print(f"[{i}/{total}] 학습 중: {market}")
        train_and_save_model(market)
        time.sleep(0.8)  # ⏱️ API Rate Limit 방지

    print("\n✅ 전체 학습 완료")
    print(f"📁 실패한 코인은 logs/failed_markets.log 에 기록됨")

if __name__ == "__main__":
    main()