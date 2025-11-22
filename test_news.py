from news_handler import is_trading_safe, fetch_news_data
import datetime

print("--- TESTING NEWS HANDLER ---")

# 1. Fetch and Print All News
print("\nFetching all news for this week...")
events = fetch_news_data()
print(f"Found {len(events)} events.")

print("\n--- UPCOMING HIGH IMPACT EVENTS (USD) ---")
for event in events:
    if event['impact'] == 'High' and event['country'] == 'USD':
        print(f"{event['datetime']} | {event['country']} | {event['title']}")

# 2. Check Safety
print("\n--- SAFETY CHECK ---")
safe, reason = is_trading_safe()
print(f"Is Trading Safe? {safe}")
print(f"Reason: {reason}")
