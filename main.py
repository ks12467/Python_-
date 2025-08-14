import time
import json
from data_loader import get_all_krw_markets, get_ohlcv
from analyzer import get_rsi, get_macd_signal, is_uptrend
from predictor import predict_next
from utils import send_discord_message
from risk import calculate_atr, calculate_stop_loss, calculate_take_profit

def main():
    message = "📊 윈터의 코인 예측 리포트 💙\n\n"
    markets = get_all_krw_markets()
    filtered = []
    recommended = []

    # ✅ 강화학습 추천 결과 불러오기
    try:
        with open("recommended_by_rl.json") as f:
            rl_recommendations = json.load(f)
    except Exception:
        rl_recommendations = {}

    for market in markets:
        try:
            time.sleep(0.25)  # ✅ API 요청 간 250ms 대기
            df = get_ohlcv(market)
            if len(df) < 50:
                continue

            rsi = get_rsi(df)
            macd, signal = get_macd_signal(df)

            if not (45 < rsi < 65 and macd > signal and is_uptrend(df)):
                continue

            filtered.append(market)
            current = df['close'].iloc[-1]

            try:
                predicted = predict_next(df, market)
            except FileNotFoundError as e:
                print(f"❌ {market} 예측 불가: {e}")
                continue

            atr = calculate_atr(df)
            stop_loss = calculate_stop_loss(current, atr)
            take_profit = calculate_take_profit(current, atr)
            expected_return = (predicted - current) / current * 100

            if expected_return >= 3:
                recommended.append({
                    "market": market,
                    "current": current,
                    "predicted": predicted,
                    "expected_return": expected_return,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                })

        except Exception as e:
            print(f"❌ {market} 오류: {e}")
            continue

    if filtered:
        message += "📈 상승 추세인 코인들 (LSTM 기반)\n"
        for m in filtered:
            message += f"👉 {m}\n"
        message += "\n"

    if recommended:
        message += "💡 LSTM 기반 추천 코인 (예측 수익률 ≥ 3%)\n"
        for r in recommended:
            message += (
                f"✅ {r['market']}\n"
                f"현재가: {r['current']:.0f} → 예측가: {r['predicted']:.0f} "
                f"(+{r['expected_return']:.2f}%)\n"
                f"익절가: {r['take_profit']:.0f}, 손절가: {r['stop_loss']:.0f}\n\n"
            )
    else:
        message += "😭 조건에 맞는 LSTM 추천 코인이 없어… 다음에 다시 보자~\n"

    if rl_recommendations:
        message += "🤖 강화학습 추천 종목\n"
        for market, reward in rl_recommendations.items():
            message += f"🔥 {market} (보상: {reward:.2f})\n"
    else:
        message += "😔 강화학습 추천 종목 없음\n"

    send_discord_message(message)
    print(message)

if __name__ == "__main__":
    main()