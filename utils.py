#utils.py
import requests
from config import DISCORD_WEBHOOK_URL

def send_discord_message(content):
    try:
        res = requests.post(DISCORD_WEBHOOK_URL, json={"content": content})
        if res.status_code != 204:
            print(f"❌ 전송 실패: {res.status_code}")
    except Exception as e:
        print(f"❌ 디스코드 오류: {e}")