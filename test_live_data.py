import pandas as pd
import time
from mt5_handler import initialize_mt5, get_mt5_data
import yfinance as yf

print("--- TESTING LIVE DATA FETCHING ---")

if not initialize_mt5():
    print("MT5 Init Failed")
    exit()

SYMBOL = "XAUUSD"
LOOKBACK_CANDLES = 100

# 1. Fetch XAUUSD
print(f"Fetching {SYMBOL}...")
df_5m = get_mt5_data(SYMBOL, "5m", LOOKBACK_CANDLES)
if df_5m is None:
    print("Failed to fetch XAUUSD")
    exit()

print(f"XAUUSD Data: {df_5m.shape}")

# 2. Fetch DXY
print("Fetching USDX (DXY)...")
df_dxy = get_mt5_data("USDX", "5m", LOOKBACK_CANDLES)
if df_dxy is not None:
    # Ensure indices are compatible (timezone naive)
    if df_5m.index.tz is not None:
        df_5m.index = df_5m.index.tz_localize(None)
    if df_dxy.index.tz is not None:
        df_dxy.index = df_dxy.index.tz_localize(None)
        
    # Merge using reindex/ffill to handle slight time diffs
    df_5m['DXY'] = df_dxy['Close'].reindex(df_5m.index, method='ffill')
    print(f"DXY Data Merged. Last Value: {df_5m['DXY'].iloc[-1]}")
else:
    print("Failed to fetch USDX")
    df_5m['DXY'] = 0.0

# 3. Fetch US10Y
print("Fetching ^TNX (US10Y)...")
try:
    us10y_data = yf.download('^TNX', period='5d', interval='5m', progress=False)
    print(f"US10Y Raw Shape: {us10y_data.shape}")
    print(f"US10Y Columns: {us10y_data.columns}")
    
    if not us10y_data.empty:
        if us10y_data.index.tz is not None:
            us10y_data.index = us10y_data.index.tz_localize(None)
            
        if isinstance(us10y_data.columns, pd.MultiIndex):
            # Check if 'Close' is top level or second level
            # yfinance structure: (Price, Ticker) -> ('Close', '^TNX')
            try:
                us10y_close = us10y_data['Close']['^TNX']
            except KeyError:
                # Maybe it's just 'Close' if single ticker
                us10y_close = us10y_data['Close']
        else:
            us10y_close = us10y_data['Close']
        
        aligned_us10y = us10y_close.reindex(df_5m.index, method='ffill')
        df_5m['US10Y'] = aligned_us10y
        print(f"US10Y Data Merged. Last Value: {df_5m['US10Y'].iloc[-1]}")
    else:
        print("US10Y Data Empty")
        df_5m['US10Y'] = 0.0
except Exception as e:
    print(f"Error fetching US10Y: {e}")
    df_5m['US10Y'] = 0.0

print("\nFinal DataFrame Head:")
print(df_5m[['Close', 'DXY', 'US10Y']].tail())
