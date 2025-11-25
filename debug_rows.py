"""Debug script to check how many rows are left after indicator calculation"""
from mt5_handler import initialize_mt5, get_mt5_data
from smc_indicators import add_all_smc_indicators
import pandas_ta as ta

if initialize_mt5():
    print("Fetching 80 candles...")
    df = get_mt5_data('XAUUSD', '5m', count=80)
    print(f"Initial rows: {len(df)}")
    
    # Add basic indicators
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    print(f"After basic indicators: {len(df)}")
    
    # Add SMC
    df = add_all_smc_indicators(df)
    print(f"After SMC indicators: {len(df)}")
    
    # Drop NaN
    df.dropna(inplace=True)
    print(f"After dropna: {len(df)}")
    print(f"Need 60 for sequence, have {len(df)}")
    print(f"Enough data: {len(df) >= 60}")
