import requests
import time

# --- CONFIGURATION ---
# Option 1: CallMeBot (Free, Personal)
# Get API Key: Send "I allow callmebot to send me messages" to +34 644 10 55 84 on WhatsApp
CALLMEBOT_PHONE = "+919360509436" # User provided
CALLMEBOT_API_KEY = "" # Get this from the bot

# Option 2: Twilio (Paid, Professional)
TWILIO_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_FROM = "whatsapp:+14155238886"
TWILIO_TO = "whatsapp:+91..."

# Option 3: Telegram (Free, Reliable, Recommended)
# 1. Search for @BotFather on Telegram
# 2. Send /newbot and follow instructions to get TOKEN
# 3. Search for @userinfobot to get your CHAT ID
TELEGRAM_BOT_TOKEN = "8180899334:AAEFt1nt9ECI5DD8aS7mBsRtAP9Y7nLcjVg" 
TELEGRAM_CHAT_ID = "1246626854"

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
