import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib
import os

# --- CONFIGURATION ---
SEQ_LEN = 120  # Increased from 60 to 120 (10 hours of 5m data)
FUTURE_TARGET = 1
PROCESSED_DATA_DIR = 'processed_data'

# Data paths
CSV_FILE_PATH_5M = os.path.join(PROCESSED_DATA_DIR, 'XAUUSD_M5_processed.csv')
CSV_FILE_PATH_15M = os.path.join(PROCESSED_DATA_DIR, 'XAUUSD_M15_processed.csv')
CSV_FILE_PATH_1H = os.path.join(PROCESSED_DATA_DIR, 'XAUUSD_H1_processed.csv')
CSV_FILE_PATH_4H = os.path.join(PROCESSED_DATA_DIR, 'XAUUSD_H4_processed.csv')
CSV_FILE_PATH_DAILY = os.path.join(PROCESSED_DATA_DIR, 'XAUUSD_Daily_processed.csv')

# --- STEP 1: DATA FUNCTIONS ---

def load_data(csv_path):
    if os.path.exists(csv_path):
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path, parse_dates=['DateTime'], index_col='DateTime')
        print(f"Data Loaded. Shape: {df.shape}")
        return df
    else:
        print(f"Warning: {csv_path} not found.")
        return None

def add_smc_indicators(df):
    """SMC features"""
    df['Swing_High'] = df['High'].rolling(window=5, center=True).max() == df['High']
    df['Swing_Low'] = df['Low'].rolling(window=5, center=True).min() == df['Low']
    
    df['Last_Swing_High'] = df['High'].where(df['Swing_High']).ffill()
    df['Last_Swing_Low'] = df['Low'].where(df['Swing_Low']).ffill()
    
    df['Dist_to_High'] = df['Last_Swing_High'] - df['Close']
    df['Dist_to_Low'] = df['Close'] - df['Last_Swing_Low']
    
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def add_technical_indicators(df, prefix=''):
    """Add technical indicators"""
    # Trend
    df[f'{prefix}EMA_50'] = ta.ema(df['Close'], length=50)
    df[f'{prefix}EMA_200'] = ta.ema(df['Close'], length=200)
    
    # Momentum
    df[f'{prefix}RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    if isinstance(macd, pd.DataFrame):
        df[f'{prefix}MACD'] = macd.iloc[:, 0]
    else:
        df[f'{prefix}MACD'] = macd
    
    # Volatility
    df[f'{prefix}ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Bollinger Bands
    bb = ta.bbands(df['Close'], length=20)
    if bb is not None:
        df[f'{prefix}BB_UPPER'] = bb.iloc[:, 0]
        df[f'{prefix}BB_LOWER'] = bb.iloc[:, 2]

    # SMC (only for main timeframe)
    if prefix == '':
        df = add_smc_indicators(df)
        
    # Volume features
    if 'Volume' in df.columns:
        df[f'{prefix}Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df[f'{prefix}Volume_Ratio'] = df['Volume'] / df[f'{prefix}Volume_MA']

    # Log Returns
    df[f'{prefix}Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Trend Direction (for HTF)
    if prefix != '':
        df[f'{prefix}Trend'] = np.where(df['Close'] > df[f'{prefix}EMA_50'], 1, -1)

    df.dropna(inplace=True)
    return df

def align_higher_timeframe(df_main, df_htf, prefix):
    """Align higher timeframe data to main timeframe"""
    print(f"Aligning {prefix} data...")
    
    # Select indicator columns
    htf_cols = [col for col in df_htf.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    df_htf_selected = df_htf[htf_cols].copy()
    
    # Rename with prefix
    df_htf_selected.columns = [f'{prefix}{col}' for col in df_htf_selected.columns]
    
    model = Sequential()
    
    # Layer 1: LSTM (increased units)
    model.add(LSTM(units=256, return_sequences=True, input_shape=input_shape, 
                   kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    # Layer 2: LSTM
    model.add(LSTM(units=256, return_sequences=True, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    
    # Layer 3: LSTM (new layer)
    model.add(LSTM(units=128, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    # Dense layers
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    
    # Output
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ENHANCED MODEL TRAINING WITH 55-YEAR DATASET")
    print("="*70)
    
    # 1. Load Main Timeframe (5m)
    print("\n--- Loading Data ---")
    df_5m = load_data(CSV_FILE_PATH_5M)
    if df_5m is None:
        exit()
    
    # 2. Load Higher Timeframes
    df_15m = load_data(CSV_FILE_PATH_15M)
    df_1h = load_data(CSV_FILE_PATH_1H)
    df_4h = load_data(CSV_FILE_PATH_4H)
    df_daily = load_data(CSV_FILE_PATH_DAILY)
    
    # 3. Add indicators
    print("\n--- Adding Technical Indicators ---")
    df_5m = add_technical_indicators(df_5m, prefix='')
    
    if df_15m is not None:
        df_15m = add_technical_indicators(df_15m, prefix='')
        df_5m = align_higher_timeframe(df_5m, df_15m, 'HTF15_')
    
    if df_1h is not None:
        df_1h = add_technical_indicators(df_1h, prefix='')
        df_5m = align_higher_timeframe(df_5m, df_1h, 'HTF1H_')
    
    if df_4h is not None:
        df_4h = add_technical_indicators(df_4h, prefix='')
        df_5m = align_higher_timeframe(df_5m, df_4h, 'HTF4H_')
    
    if df_daily is not None:
        df_daily = add_technical_indicators(df_daily, prefix='')
        df_5m = align_higher_timeframe(df_5m, df_daily, 'HTFD_')
    
    # 4. Preprocess
    print("\n--- Preprocessing Data ---")
    X, y, scaler, target_scaler, features = preprocess_data(df_5m)

    # 5. Time-based split (90% train, 10% test - last 6 months)
    split = int(len(X) * 0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\n--- Data Split ---")
    print(f"Training Data: {X_train.shape[0]:,} sequences")
    print(f"Testing Data: {X_test.shape[0]:,} sequences")
    print(f"Sequence Length: {SEQ_LEN} candles")
    print(f"Features per candle: {X_train.shape[2]}")

    # 6. Save Scalers
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')
    print("\n✓ Scalers saved")

    # 7. Build Model
    print("\n--- Building Enhanced Model ---")
    model = build_model((X_train.shape[1], X_train.shape[2]))
    print(f"Total Parameters: {model.count_params():,}")

    # 8. Callbacks
    checkpoint = ModelCheckpoint("best_xauusd_model.keras", save_best_only=True, 
                                monitor='val_loss', mode='min', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 
                                  min_lr=0.00001, verbose=1)

    # 9. Train
    print("\n--- Starting Training ---")
    print("This may take 30-60 minutes...")
    history = model.fit(
        X_train, y_train,
        epochs=30, 
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"Model saved as: best_xauusd_model.keras")