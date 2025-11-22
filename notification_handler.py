import requests
import time

import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
# Option 1: CallMeBot (Free, Personal)
# Get API Key: Send "I allow callmebot to send me messages" to +34 644 10 55 84 on WhatsApp
CALLMEBOT_PHONE = os.getenv("CALLMEBOT_PHONE")
CALLMEBOT_API_KEY = os.getenv("CALLMEBOT_API_KEY")

# Option 2: Twilio (Paid, Professional)
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")
TWILIO_TO = os.getenv("TWILIO_TO")

# Option 3: Telegram (Free, Reliable, Recommended)
# 1. Search for @BotFather on Telegram
# 2. Send /newbot and follow instructions to get TOKEN
# 3. Search for @userinfobot to get your CHAT ID
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    try:
        # Telegram API URL
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        response = requests.post(url, data=data)
        if response.status_code == 200:
            print("Notification sent via Telegram!")
            return True
        else:
            print(f"Telegram Failed: {response.text}")
            return False
    except Exception as e:
        print(f"Error sending Telegram: {e}")
        return False

def send_whatsapp_message(message):
    """
    Sends a notification using the configured provider.
    Prioritizes Telegram if configured.
    """
    # Try Telegram first (Most Reliable)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        return send_telegram_message(message)

    # Try CallMeBot
    if CALLMEBOT_PHONE and CALLMEBOT_API_KEY:
        return send_callmebot(message)
    
    # Try Twilio
    if TWILIO_SID and TWILIO_AUTH_TOKEN:
        return send_twilio(message)
        
    print("No notification service configured.")
    return False

def send_callmebot(message):
    try:
        # URL Encode message
        encoded_msg = requests.utils.quote(message)
        url = f"https://api.callmebot.com/whatsapp.php?phone={CALLMEBOT_PHONE}&text={encoded_msg}&apikey={CALLMEBOT_API_KEY}"
        
        response = requests.get(url)
        if response.status_code == 200:
            print("WhatsApp sent via CallMeBot!")
            return True
        else:
            print(f"CallMeBot Failed: {response.text}")
            return False
    except Exception as e:
        print(f"Error sending WhatsApp: {e}")
        return False

def send_twilio(message):
    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        
        msg = client.messages.create(
            body=message,
            from_=TWILIO_FROM,
            to=TWILIO_TO
        )
        print(f"WhatsApp sent via Twilio! SID: {msg.sid}")
        return True
    except Exception as e:
        print(f"Twilio Error: {e}")
        return False
