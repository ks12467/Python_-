# train_rl.py

import os
import time
import json
from reinforcement.environment import TradingEnvironment
from reinforcement.agent import DQNAgent
from data_loader import get_ohlcv, get_all_krw_markets

# 폴더 생성
os.makedirs("logs", exist_ok=True)
RECOMMEND_PATH = "recommended_by_rl.json"
FAILED_LOG_PATH = "logs/rl_failed_markets.log"

# 파라미터
EPISODES = 5
REWARD_THRESHOLD = 50  # 보상 기준으로 추천할지 판단
DELAY = 0.3  # API 제한 방지

def train_rl_model(market, episodes=EPISODES):
    df = get_ohlcv(market, count=200, unit=5)
    if len(df) < 50:
        raise ValueError("❌ 데이터 부족")

    prices = df['close'].values
    env = TradingEnvironment(prices)
    agent = DQNAgent(state_size=10, action_size=3)

    total_reward = 0
    for e in range(episodes):
        state = env.reset()
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        agent.replay()
    return total_reward

def main():
    markets = get_all_krw_markets()
    results = {}
    success = 0

    print(f"🤖 강화학습 시작: {len(markets)}개 코인 대상\n")

    for market in markets:
        try:
            reward = train_rl_model(market)
            print(f"✅ {market} 보상: {reward:.2f}")
            if reward >= REWARD_THRESHOLD:
                results[market] = reward
                success += 1
        except Exception as e:
            print(f"❌ {market} 실패: {e}")
            with open(FAILED_LOG_PATH, "a") as f:
                f.write(f"{market}: {e}\n")
        time.sleep(DELAY)

    # 보상 기준 상위 코인 저장
    sorted_result = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    with open(RECOMMEND_PATH, "w") as f:
        json.dump(sorted_result, f, indent=2)

    print(f"\n🎯 추천 종목 저장 완료: {success}개")
    print(f"📁 파일: {RECOMMEND_PATH}")
    print(f"📄 실패 로그: {FAILED_LOG_PATH}")

if __name__ == "__main__":
    main()