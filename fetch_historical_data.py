import MetaTrader5 as mt5
import pandas as pd
import yfinance as yf
import datetime
import time

# --- CONFIGURATION ---
COUNT = 50000  # Reduced to avoid MT5 limits
SYMBOL_XAU = "XAUUSD"
SYMBOL_DXY = "USDX"
SYMBOL_US10Y = "^TNX"

def initialize_mt5():
    if not mt5.initialize():
        print("MT5 Initialize Failed")
        return False
    return True

def get_mt5_data(symbol, count):
    print(f"Fetching {count} candles for {symbol} from MT5...")
    
    # Ensure symbol is selected
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
        return None
        
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, count)
    if rates is None or len(rates) == 0:
        print(f"Failed to fetch {symbol}. Error: {mt5.last_error()}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    df.set_index('Date', inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def main():
    if not initialize_mt5():
        return

    # 1. Fetch XAUUSD (Target)
    df_xau = get_mt5_data(SYMBOL_XAU, COUNT)
    if df_xau is None:
        return
    print(f"XAUUSD Data: {df_xau.shape}")

    # 2. Fetch DXY (Feature)
    df_dxy = get_mt5_data(SYMBOL_DXY, COUNT)
    if df_dxy is not None:
        # Rename to avoid collision
        df_dxy = df_dxy[['Close']].rename(columns={'Close': 'DXY'})
        # Merge
        df_merged = df_xau.join(df_dxy, how='left')
    else:
        print("Warning: DXY missing. Filling with 0.")
        df_merged = df_xau.copy()
        df_merged['DXY'] = 0.0

    # 3. Fetch US10Y (Feature) - DAILY Data upsampled
    print(f"Fetching Daily US10Y data from YFinance...")
    # Fetch 2 years to cover the 1 year of 5m data
    us10y_data = yf.download(SYMBOL_US10Y, period='2y', interval='1d', progress=False)
    
    if not us10y_data.empty:
        # Handle MultiIndex
        if isinstance(us10y_data.columns, pd.MultiIndex):
            try:
                us10y_close = us10y_data['Close'][SYMBOL_US10Y]
            except KeyError:
                us10y_close = us10y_data['Close']
        else:
            us10y_close = us10y_data['Close']
            
        # Rename
        us10y_close = us10y_close.rename('US10Y')
        
        # Remove timezone if present to match MT5
        if us10y_close.index.tz is not None:
            us10y_close.index = us10y_close.index.tz_localize(None)

        # Merge and Upsample
        # We join the daily data to the 5m index. 
        # This will put the daily close at 00:00 of that day (or close time).
        # Then we ffill to propagate that value forward for the rest of the day.
        df_merged = df_merged.join(us10y_close, how='left')
        df_merged['US10Y'] = df_merged['US10Y'].ffill()
        
    else:
        print("Warning: US10Y missing. Filling with 0.")
        df_merged['US10Y'] = 0.0

    # 4. Cleanup
    print(f"Shape before dropna: {df_merged.shape}")
    print(f"NaN counts:\n{df_merged.isna().sum()}")
    
    df_merged.fillna(method='ffill', inplace=True) # Forward fill first
    df_merged.dropna(inplace=True)
    print(f"Final Dataset Shape: {df_merged.shape}")
    print(df_merged.head())
    print(df_merged.tail())

    # 5. Save
    csv_path = "training_data.csv"
    df_merged.to_csv(csv_path)
    print(f"Saved to {csv_path}")
    
    mt5.shutdown()

if __name__ == "__main__":
    main()
