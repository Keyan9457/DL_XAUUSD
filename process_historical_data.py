import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob

# --- CONFIGURATION ---
DATA_DIR = r"C:\Users\Admin\Documents\DL_XAUUSD\XAUUSD_HISTORICAL_DATA"
OUTPUT_DIR = r"C:\Users\Admin\Documents\DL_XAUUSD\processed_data"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_mt_csv(filepath):
    """
    Load MetaTrader CSV format with separate DATE and TIME columns
    """
    print(f"\nLoading: {os.path.basename(filepath)}")
    
    try:
        # Read CSV with specific column names
        df = pd.read_csv(
            filepath,
            sep='\t',  # MetaTrader uses tab separator
            skiprows=0,
            names=['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD'],
            skipinitialspace=True
        )
        
        # Remove header row if it exists
        if df['DATE'].iloc[0] == '<DATE>':
            df = df.iloc[1:]
        
        # Combine DATE and TIME into datetime
        df['DateTime'] = pd.to_datetime(
            df['DATE'].astype(str) + ' ' + df['TIME'].astype(str),
            format='%Y.%m.%d %H:%M:%S'
        )
        
        # Set datetime as index
        df.set_index('DateTime', inplace=True)
        
        # Rename columns to standard format
        df.rename(columns={
            'OPEN': 'Open',
            'HIGH': 'High',
            'LOW': 'Low',
            'CLOSE': 'Close',
            'TICKVOL': 'Volume'
        }, inplace=True)
        
        # Select only needed columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN
        df.dropna(inplace=True)
        
        print(f"  ✓ Loaded {len(df):,} bars")
        print(f"  ✓ Date Range: {df.index[0]} to {df.index[-1]}")
        print(f"  ✓ Price Range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"  ✗ Error loading file: {e}")
        return None

def validate_data(df, timeframe_name):
    """
    Validate data quality
    """
    print(f"\n  Validating {timeframe_name}...")
    
    issues = []
    
    # Check for duplicates
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate timestamps")
        df = df[~df.index.duplicated(keep='first')]
    
    # Check for invalid prices (negative or zero)
    invalid_prices = (df[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1).sum()
    if invalid_prices > 0:
        issues.append(f"Found {invalid_prices} rows with invalid prices")
        df = df[(df[['Open', 'High', 'Low', 'Close']] > 0).all(axis=1)]
    
    # Check for OHLC consistency
    invalid_ohlc = (
        (df['High'] < df['Low']) |
        (df['High'] < df['Open']) |
        (df['High'] < df['Close']) |
        (df['Low'] > df['Open']) |
        (df['Low'] > df['Close'])
    ).sum()
    if invalid_ohlc > 0:
        issues.append(f"Found {invalid_ohlc} rows with invalid OHLC relationships")
        # Filter out invalid OHLC
        df = df[
            (df['High'] >= df['Low']) &
            (df['High'] >= df['Open']) &
            (df['High'] >= df['Close']) &
            (df['Low'] <= df['Open']) &
            (df['Low'] <= df['Close'])
        ]
    
    if issues:
        print(f"  ⚠ Issues found and corrected:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  ✓ Data validation passed")
    
    return df

def process_timeframe(filepath, timeframe_name):
    """
    Process a single timeframe file
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING: {timeframe_name}")
    print(f"{'='*70}")
    
    # Load data
    df = load_mt_csv(filepath)
    if df is None:
        return None
    
    # Validate data
    df = validate_data(df, timeframe_name)
    
    # Save processed data
    output_file = os.path.join(OUTPUT_DIR, f"XAUUSD_{timeframe_name}_processed.csv")
    df.to_csv(output_file)
    print(f"\n  ✓ Saved to: {output_file}")
    print(f"  ✓ Final rows: {len(df):,}")
    
    return df

def main():
    print("\n" + "="*70)
    print("XAUUSD HISTORICAL DATA PROCESSING")
    print("="*70)
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(DATA_DIR, "XAUUSD_*.csv"))
    print(f"\nFound {len(csv_files)} CSV files")
    
    # Priority timeframes for model training
    priority_timeframes = {
        'M5': 'XAUUSD_M5_*.csv',      # Primary timeframe
        'M15': 'XAUUSD_M15_*.csv',    # HTF 1
        'H1': 'XAUUSD_H1_*.csv',      # HTF 2
        'H4': 'XAUUSD_H4_*.csv',      # HTF 3
        'Daily': 'XAUUSD_Daily_*.csv' # HTF 4
    }
    
    processed_data = {}
    
    # Process priority timeframes first
    print(f"\n{'='*70}")
    print("PROCESSING PRIORITY TIMEFRAMES")
    print(f"{'='*70}")
    
    for tf_name, pattern in priority_timeframes.items():
        matching_files = glob.glob(os.path.join(DATA_DIR, pattern))
        if matching_files:
            df = process_timeframe(matching_files[0], tf_name)
            if df is not None:
                processed_data[tf_name] = df
        else:
            print(f"\n⚠ Warning: No file found for {tf_name}")
    
    # Summary
    print(f"\n{'='*70}")
    print("PROCESSING SUMMARY")
    print(f"{'='*70}")
    
    for tf_name, df in processed_data.items():
        print(f"\n{tf_name}:")
        print(f"  Rows: {len(df):,}")
        print(f"  Date Range: {df.index[0]} to {df.index[-1]}")
        print(f"  Duration: {(df.index[-1] - df.index[0]).days} days")
        print(f"  Avg Volume: {df['Volume'].mean():.0f}")
    
    print(f"\n{'='*70}")
    print("✓ DATA PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"\nProcessed files saved to: {OUTPUT_DIR}")
    
    return processed_data

if __name__ == "__main__":
    processed_data = main()
