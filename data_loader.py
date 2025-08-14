# data_loader.py

import requests
import pandas as pd
import time

def get_ohlcv(market="KRW-BTC", count=200, unit=5, retries=3):
    url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    params = {"market": market, "count": count}
    headers = {"Accept": "application/json"}

    for _ in range(retries):
        try:
            res = requests.get(url, params=params, headers=headers)
            res.raise_for_status()
            df = pd.DataFrame(res.json())
            df = df[['candle_date_time_kst', 'trade_price', 'candle_acc_trade_volume']]
            df.columns = ['timestamp', 'close', 'volume']
            return df.sort_values(by='timestamp').reset_index(drop=True)
        except requests.exceptions.HTTPError as e:
            if res.status_code == 429:
                print(f"⏳ 429 Too Many Requests – 잠시 대기 후 재시도")
                time.sleep(2)
            else:
                raise
        except Exception as e:
            raise Exception(f"❌ {market} OHLCV 불러오기 실패: {e}")

    raise Exception(f"❌ {market} OHLCV 요청 3회 실패")

def get_all_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    res = requests.get(url).json()
    return [m['market'] for m in res if m['market'].startswith("KRW-")]