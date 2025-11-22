import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import pytz
import time

# URL for ForexFactory Weekly Calendar
NEWS_URL = "http://nfs.faireconomy.media/ff_calendar_thisweek.xml"

def fetch_news_data():
    """
    Fetches news data from ForexFactory and parses it.
    Returns a list of dictionaries containing news events.
    """
    try:
        response = requests.get(NEWS_URL)
        if response.status_code != 200:
            print(f"Failed to fetch news: {response.status_code}")
            return []

        root = ET.fromstring(response.content)
        news_events = []

        for event in root.findall('event'):
            title = event.find('title').text
            country = event.find('country').text
            date_str = event.find('date').text
            time_str = event.find('time').text
            impact = event.find('impact').text
            
            # Combine date and time
            # Format: 11-22-2025 and 10:00am
            try:
                dt_str = f"{date_str} {time_str}"
                # ForexFactory XML times are usually in EST/EDT (New York time) or GMT depending on server, 
                # but often it's best to treat them as naive or check headers. 
                # However, the feed usually returns times relative to the request or standard EST.
                # Let's assume standard format parsing first.
                
                # Note: The feed time format can be tricky (e.g. "1:30pm").
                event_dt = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p")
                
                # Assuming the feed is in UTC or EST. 
                # A safer bet for this specific feed is usually that it matches the website's default or is UTC.
                # Let's assume it's UTC for now to be safe, or we can check against current time.
                # Actually, FF XML is often loosely defined. Let's treat as naive and compare carefully.
                # Better approach: Convert everything to UTC if possible.
                
                news_events.append({
                    'title': title,
                    'country': country,
                    'datetime': event_dt,
                    'impact': impact
                })
            except ValueError:
                continue

        return news_events

    except Exception as e:
        print(f"Error parsing news: {e}")
        return []

def is_trading_safe(symbol="XAUUSD", minutes_before=60, minutes_after=60):
    """
    Checks if it is safe to trade based on high-impact news.
    
    Args:
        symbol (str): The symbol being traded (e.g., "XAUUSD").
        minutes_before (int): Minutes to stop trading before news.
        minutes_after (int): Minutes to stop trading after news.
        
    Returns:
        tuple: (bool, str) -> (is_safe, reason)
    """
    events = fetch_news_data()
    if not events:
        print("Warning: Could not fetch news. Assuming safe (or unsafe depending on preference).")
        return True, "No news data"

    # Filter for relevant currencies
    # For XAUUSD, we care about USD and potentially major global news.
    relevant_currencies = ["USD"]
    if "EUR" in symbol: relevant_currencies.append("EUR")
    if "GBP" in symbol: relevant_currencies.append("GBP")
    if "JPY" in symbol: relevant_currencies.append("JPY")
    
    # Current time (Naive, matching the XML parse)
    # IMPORTANT: We need to align timezones. 
    # If the XML is EST, we need current EST.
    # Let's try to detect offset or use a library if needed. 
    # For simplicity in this snippet, we'll use system time but be aware of offset.
    # A common issue with FF XML is timezone. 
    # Let's assume the XML date is accurate to the day and time is local to the exchange.
    
    # WORKAROUND: We will use a relative check.
    now = datetime.now()
    
    # We need to know the timezone of the XML. 
    # Empirically, FF XML is often EST (UTC-5) or EDT (UTC-4).
    # Let's assume New York time.
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny).replace(tzinfo=None) # Naive NY time for comparison
    
    # Actually, let's just look at the difference.
    
    for event in events:
        if event['impact'] == 'High' and event['country'] in relevant_currencies:
            # Check time difference
            # We'll assume the event['datetime'] is in NY time (standard for FF)
            
            time_diff = (event['datetime'] - now_ny).total_seconds() / 60
            
            # Check if we are in the danger zone
            # Danger zone: from (now - minutes_after) to (now + minutes_before)
            # Wait, logic:
            # If event is in the future (time_diff > 0): is it within minutes_before?
            # If event is in the past (time_diff < 0): is it within minutes_after?
            
            if 0 <= time_diff <= minutes_before:
                return False, f"Upcoming High Impact News: {event['title']} in {int(time_diff)} mins"
            
            if -minutes_after <= time_diff < 0:
                return False, f"Recent High Impact News: {event['title']} was {int(abs(time_diff))} mins ago"

    return True, "No relevant high-impact news nearby"

if __name__ == "__main__":
    # Test run
    safe, reason = is_trading_safe()
    print(f"Trading Safe: {safe}")
    print(f"Reason: {reason}")
