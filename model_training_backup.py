import MetaTrader5 as mt5
import pandas as pd
import yfinance as yf
import datetime
import time
import os

# --- CONFIGURATION ---
SYMBOL_XAU = "XAUUSD"
SYMBOL_DXY = "USDX"
SYMBOL_US10Y = "^TNX"

# Timeframes to fetch
TIMEFRAMES = {
    '1m': mt5.TIMEFRAME_M1,
    '3m': mt5.TIMEFRAME_M3,
    '5m': mt5.TIMEFRAME_M5,
    '15m': mt5.TIMEFRAME_M15,
    '30m': mt5.TIMEFRAME_M30,
    '1h': mt5.TIMEFRAME_H1,
    '4h': mt5.TIMEFRAME_H4,
    '1d': mt5.TIMEFRAME_D1
}

def initialize_mt5():
    if not mt5.initialize():
        print("MT5 Initialize Failed")
        return False
    print("MT5 Initialized Successfully")
    return True

def get_max_mt5_data(symbol, timeframe, max_bars=100000):
    """
    Fetch maximum available historical data from MT5
    MT5 has limits, so we'll fetch in chunks if needed
    """
    print(f"\nFetching maximum data for {symbol} on {timeframe}...")
    
    # Ensure symbol is selected
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
        return None
    
    # Try to fetch maximum bars
    # MT5 typically limits to 100,000 bars per request
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, max_bars)
    
    if rates is None or len(rates) == 0:
        print(f"Failed to fetch {symbol}. Error: {mt5.last_error()}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={
        'time': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    }, inplace=True)
    df.set_index('Date', inplace=True)
    
    print(f"  ✓ Fetched {len(df)} bars")
    print(f"  ✓ Date Range: {df.index[0]} to {df.index[-1]}")
    
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def fetch_correlated_assets(start_date, end_date):
    """
    Fetch DXY and US10Y data for the given date range
    """
    print(f"\n--- Fetching Correlated Assets ---")
    
    # 1. DXY from MT5
    print("Fetching DXY from MT5...")
    df_dxy = get_max_mt5_data(SYMBOL_DXY, mt5.TIMEFRAME_M5, 100000)
    
    # 2. US10Y from YFinance (5-minute data)
    print("\nFetching US10Y from YFinance...")
    try:
        # Calculate days difference
        days_diff = (end_date - start_date).days + 1
        
        # YFinance limits: max 60 days for 5m data
        if days_diff > 60:
            print(f"  Warning: Requested {days_diff} days, but YFinance limits 5m data to 60 days")
            print(f"  Fetching last 60 days only...")
            us10y_data = yf.download(SYMBOL_US10Y, period='60d', interval='5m', progress=False)
        else:
            us10y_data = yf.download(SYMBOL_US10Y, start=start_date, end=end_date, interval='5m', progress=False)
        
        if not us10y_data.empty:
            # Handle MultiIndex
            if isinstance(us10y_data.columns, pd.MultiIndex):
                try:
                    us10y_close = us10y_data['Close'][SYMBOL_US10Y]
                except KeyError:
                    us10y_close = us10y_data['Close']
            else:
                us10y_close = us10y_data['Close']
            
            # Remove timezone
            if us10y_close.index.tz is not None:
                us10y_close.index = us10y_close.index.tz_localize(None)
            
            df_us10y = pd.DataFrame({'US10Y': us10y_close})
            print(f"  ✓ Fetched {len(df_us10y)} bars")
            print(f"  ✓ Date Range: {df_us10y.index[0]} to {df_us10y.index[-1]}")
        else:
            df_us10y = None
            print("  ✗ No US10Y data available")
    except Exception as e:
        print(f"  ✗ Error fetching US10Y: {e}")
        df_us10y = None
    
    return df_dxy, df_us10y

def merge_data(df_main, df_dxy, df_us10y):
    """
    Merge main data with correlated assets
    """
    print("\n--- Merging Data ---")
    
    # Start with main data
    df_merged = df_main.copy()
    
    # Merge DXY
    if df_dxy is not None:
        df_dxy_close = df_dxy[['Close']].rename(columns={'Close': 'DXY'})
        df_merged = df_merged.join(df_dxy_close, how='left')
        print(f"  ✓ Merged DXY")
    else:
        df_merged['DXY'] = 0.0
        print(f"  ⚠ DXY missing, filled with 0.0")
    
    # Merge US10Y
    if df_us10y is not None:
        df_merged = df_merged.join(df_us10y, how='left')
        print(f"  ✓ Merged US10Y")
    else:
        df_merged['US10Y'] = 0.0
        print(f"  ⚠ US10Y missing, filled with 0.0")
    
    # Forward fill missing values
    print(f"\nShape before cleanup: {df_merged.shape}")
    print(f"NaN counts:\n{df_merged.isna().sum()}")
    
    df_merged.fillna(method='ffill', inplace=True)
    df_merged.fillna(method='bfill', inplace=True)  # Backfill any remaining
    df_merged.dropna(inplace=True)
    
    print(f"\nFinal Dataset Shape: {df_merged.shape}")
    
    return df_merged

def main():
    if not initialize_mt5():
        return
    
    print("\n" + "="*60)
    print("COMPREHENSIVE HISTORICAL DATA DOWNLOAD")
    print("="*60)
    
    # Fetch data for each timeframe
    all_data = {}
    
    for tf_name, tf_code in TIMEFRAMES.items():
        print(f"\n{'='*60}")
        print(f"TIMEFRAME: {tf_name}")
        print(f"{'='*60}")
        
        # Fetch XAUUSD
        df_xau = get_max_mt5_data(SYMBOL_XAU, tf_code, 100000)
        
        if df_xau is None or len(df_xau) == 0:
            print(f"  ✗ Skipping {tf_name} - no data available")
            continue
        
        # Get date range
        start_date = df_xau.index[0]
        end_date = df_xau.index[-1]
        
        # Fetch correlated assets (only for 5m to save time)
        if tf_name == '5m':
            df_dxy, df_us10y = fetch_correlated_assets(start_date, end_date)
            df_merged = merge_data(df_xau, df_dxy, df_us10y)
        else:
            # For other timeframes, just use XAUUSD data
            df_merged = df_xau.copy()
            df_merged['DXY'] = 0.0
            df_merged['US10Y'] = 0.0
        
        # Save to CSV
        filename = f"training_data_{tf_name}.csv"
        df_merged.to_csv(filename)
        print(f"\n  ✓ Saved to {filename}")
        print(f"  ✓ Total Rows: {len(df_merged)}")
        print(f"  ✓ Columns: {list(df_merged.columns)}")
        
        all_data[tf_name] = df_merged
        
        # Small delay to avoid overwhelming MT5
        time.sleep(1)
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for tf_name, df in all_data.items():
        print(f"{tf_name:>5}: {len(df):>7} bars | {df.index[0]} to {df.index[-1]}")
    
    mt5.shutdown()
    print("\n✓ All downloads complete!")

if __name__ == "__main__":
    main()
