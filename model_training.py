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
SEQ_LEN = 120  # 10 hours of 5m data
FUTURE_TARGET = 1
PROCESSED_DATA_DIR = 'processed_data'

# Data paths
CSV_FILE_PATH_5M = os.path.join(PROCESSED_DATA_DIR, 'XAUUSD_M5_processed.csv')
CSV_FILE_PATH_15M = os.path.join(PROCESSED_DATA_DIR, 'XAUUSD_M15_processed.csv')
CSV_FILE_PATH_1H = os.path.join(PROCESSED_DATA_DIR, 'XAUUSD_H1_processed.csv')
CSV_FILE_PATH_4H = os.path.join(PROCESSED_DATA_DIR, 'XAUUSD_H4_processed.csv')

# --- DATA FUNCTIONS ---

def load_data(csv_path):
    if os.path.exists(csv_path):
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path, parse_dates=['DateTime'], index_col='DateTime')
        print(f"  ✓ Loaded {len(df):,} rows")
        return df
    else:
        print(f"  ✗ {csv_path} not found")
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
    df[f'{prefix}EMA_50'] = ta.ema(df['Close'], length=50)
    df[f'{prefix}EMA_200'] = ta.ema(df['Close'], length=200)
    
    df[f'{prefix}RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    if isinstance(macd, pd.DataFrame):
        df[f'{prefix}MACD'] = macd.iloc[:, 0]
    else:
        df[f'{prefix}MACD'] = macd
    
    df[f'{prefix}ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    bb = ta.bbands(df['Close'], length=20)
    if bb is not None:
        df[f'{prefix}BB_UPPER'] = bb.iloc[:, 0]
        df[f'{prefix}BB_LOWER'] = bb.iloc[:, 2]

    if prefix == '':
        df = add_smc_indicators(df)
        
    if 'Volume' in df.columns:
        df[f'{prefix}Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df[f'{prefix}Volume_Ratio'] = df['Volume'] / df[f'{prefix}Volume_MA']

    df[f'{prefix}Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    df.dropna(inplace=True)
    return df

def align_higher_timeframe(df_main, df_htf, prefix):
    """Align higher timeframe data"""
    print(f"  Aligning {prefix}...")
    
    htf_cols = [col for col in df_htf.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    df_htf_selected = df_htf[htf_cols].copy()
    df_htf_selected.columns = [f'{prefix}{col}' for col in df_htf_selected.columns]
    
    df_merged = df_main.join(df_htf_selected, how='left')
    df_merged.fillna(method='ffill', inplace=True)
    
    return df_merged

def preprocess_data(df, use_sampling=True):
    """Preprocess with memory-efficient sampling"""
    
    if use_sampling:
        print(f"\nApplying memory-efficient sampling...")
        print(f"  Original: {len(df):,} rows")
        
        # Use all recent data (last 2 years) + 20% of older data
        cutoff_date = df.index[-1] - pd.Timedelta(days=730)
        df_recent = df[df.index >= cutoff_date]
        df_old = df[df.index < cutoff_date]
        
        if len(df_old) > 0:
            sample_size = int(len(df_old) * 0.2)
            df_old_sampled = df_old.sample(n=sample_size, random_state=42).sort_index()
            df = pd.concat([df_old_sampled, df_recent])
        else:
            df = df_recent
        
        print(f"  Sampled: {len(df):,} rows ({len(df_recent):,} recent + {len(df) - len(df_recent):,} historical)")
    
    df['Target'] = df['Log_Return'].shift(-FUTURE_TARGET)
    df.dropna(inplace=True)

    feature_cols = [
        'Close', 'RSI', 'MACD', 'ATR', 'EMA_50', 'EMA_200', 'BB_UPPER', 'BB_LOWER',
        'Dist_to_High', 'Dist_to_Low', 'Volume_MA', 'Volume_Ratio'
    ]
    
    htf_features = [col for col in df.columns if col.startswith('HTF')]
    feature_cols.extend(htf_features)
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"\nUsing {len(feature_cols)} features")
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])
    
    target_scaler = MinMaxScaler()
    scaled_target = target_scaler.fit_transform(df[['Target']])
    
    X, y = [], []
    print(f"Creating sequences...")
    for i in range(SEQ_LEN, len(scaled_data)):
        X.append(scaled_data[i-SEQ_LEN:i])
        y.append(scaled_target[i])
        
        if i % 50000 == 0:
            print(f"  Progress: {i:,} / {len(scaled_data):,}")
        
    return np.array(X), np.array(y), scaler, target_scaler, feature_cols

def build_model(input_shape):
    """Enhanced LSTM model"""
    model = Sequential()
    
    model.add(LSTM(units=256, return_sequences=True, input_shape=input_shape, 
                   kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(LSTM(units=256, return_sequences=True, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    
    model.add(LSTM(units=128, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

# --- MAIN ---

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ENHANCED MODEL TRAINING WITH HISTORICAL DATA")
    print("="*70)
    
    # Load data
    print("\n--- Loading Data ---")
    df_5m = load_data(CSV_FILE_PATH_5M)
    if df_5m is None:
        print("\nError: M5 data not found. Run process_historical_data.py first.")
        exit()
    
    df_15m = load_data(CSV_FILE_PATH_15M)
    df_1h = load_data(CSV_FILE_PATH_1H)
    df_4h = load_data(CSV_FILE_PATH_4H)
    
    # Add indicators
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
    
    # Preprocess
    print("\n--- Preprocessing Data ---")
    X, y, scaler, target_scaler, features = preprocess_data(df_5m, use_sampling=True)

    # Split
    split = int(len(X) * 0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\n--- Data Split ---")
    print(f"Training: {X_train.shape[0]:,} sequences")
    print(f"Testing: {X_test.shape[0]:,} sequences")
    print(f"Sequence Length: {SEQ_LEN} candles")
    print(f"Features: {X_train.shape[2]}")

    # Save scalers
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')
    print("\n✓ Scalers saved")

    # Build model
    print("\n--- Building Model ---")
    model = build_model((X_train.shape[1], X_train.shape[2]))
    print(f"Parameters: {model.count_params():,}")

    # Callbacks
    checkpoint = ModelCheckpoint("best_xauusd_model.keras", save_best_only=True, 
                                monitor='val_loss', mode='min', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 
                                  min_lr=0.00001, verbose=1)

    # Train
    print("\n--- Starting Training ---")
    print("This will take 30-60 minutes...")
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
    print("Model saved as: best_xauusd_model.keras")