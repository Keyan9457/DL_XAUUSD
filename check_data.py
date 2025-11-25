"""Quick diagnostic to check MT5 data availability"""
from mt5_handler import initialize_mt5, get_mt5_data

if initialize_mt5():
    print("✅ MT5 Connected")
    
    # Try to fetch different amounts of data
    for count in [50, 80, 100, 200, 300]:
        df = get_mt5_data('XAUUSD', '5m', count=count)
        if df is not None:
            print(f"✅ {count} candles: Got {len(df)} rows")
        else:
            print(f"❌ {count} candles: Failed")
else:
    print("❌ MT5 Connection Failed")
