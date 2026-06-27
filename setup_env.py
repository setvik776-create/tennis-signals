import os

import requests


TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL = os.getenv("TELEGRAM_CHANNEL", "@tennisoraclesignal")
ENV_PATH = "/root/tennis_signals/.env"

if not TOKEN:
    raise SystemExit("Set TELEGRAM_BOT_TOKEN before running setup_env.py")

url = f"https://api.telegram.org/bot{TOKEN}/getChat?chat_id={CHANNEL}"

try:
    res = requests.get(url, timeout=20).json()
    if res.get("ok"):
        chat_id = res["result"]["id"]
        with open(ENV_PATH, "w") as f:
            f.write(f'TELEGRAM_BOT_TOKEN="{TOKEN}"\nTELEGRAM_CHAT_ID="{chat_id}"\n')
        os.chmod(ENV_PATH, 0o600)
        print(f"Sekme! Kanalo ID: {chat_id}. .env sukurtas ir apsaugotas (chmod 600).")
    else:
        print(f"Telegram API Klaida: {res}")
except Exception as e:
    print(f"Klaida: {e}")
