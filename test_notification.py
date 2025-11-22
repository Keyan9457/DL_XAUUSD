from notification_handler import send_whatsapp_message

print("Attempting to send test message...")
success = send_whatsapp_message("🔔 *TEST NOTIFICATION* 🔔\n\nYour trading bot is active and connected via Telegram!")

if success:
    print("✅ Test message sent successfully!")
else:
    print("❌ Failed to send test message.")
