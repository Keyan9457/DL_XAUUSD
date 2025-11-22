import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os

# --- CONFIGURATION ---
SEQ_LEN = 60  # Look back at the last 60 candles (5 hours)
FUTURE_TARGET = 1  # Predict the next 1 candle
CSV_FILE_PATH = 'XAUUSD_5M_20Y.csv' # Ensure this file exists in your folder

# --- STEP 1: DATA FUNCTIONS ---

def load_data(csv_path):
    if os.path.exists(csv_path):
        print(f"Loading data from {csv_path}...")
        # Ensure your CSV has 'Date' column. Adjust names if your CSV is different.
        df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    else:
        # Fallback: Download recent data for testing if CSV is missing
        import yfinance as yf
        print("CSV not found. Downloading recent sample data from YFinance...")
        df = yf.download('GC=F', period='60d', interval='5m', progress=False) 
        
        # Check for MultiIndex columns (common in new yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            print("Detected MultiIndex columns. Flattening...")
            # If level 1 is Ticker, we can drop it or just take level 0
            df.columns = df.columns.get_level_values(0)
        
        print(f"Columns after loading: {df.columns}")
        
        # YFinance columns are usually: Open, High, Low, Close, Volume
        # Ensure we have the right case
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Basic cleanup
    df.dropna(inplace=True)
    return df

def add_smc_indicators(df):
    """
    Adds simplified SMC features:
    1. Swing Highs/Lows (Fractals)
    2. Order Block Proximity (Distance to recent significant High/Low)
    """
    # Identify Swing Highs and Lows (Fractals) - Window of 5
    # A high is a high if it's higher than 2 candles before and 2 after
    df['Swing_High'] = df['High'].rolling(window=5, center=True).max() == df['High']
    df['Swing_Low'] = df['Low'].rolling(window=5, center=True).min() == df['Low']
    
    # Forward fill the last known Swing High/Low to simulate "Order Block" zones
    # In a real SMC system, this would be more complex (mitigation, etc.)
    df['Last_Swing_High'] = df['High'].where(df['Swing_High']).ffill()
    df['Last_Swing_Low'] = df['Low'].where(df['Swing_Low']).ffill()
    
    # Feature: Distance to Last Swing High/Low (Normalized by ATR later implicitly)
    # If Price is close to Last Swing High -> Potential Bearish OB
    # If Price is close to Last Swing Low -> Potential Bullish OB
    df['Dist_to_High'] = df['Last_Swing_High'] - df['Close']
    df['Dist_to_Low'] = df['Close'] - df['Last_Swing_Low']
    
    # Fill NaNs created by rolling/shifting
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True) # For initial rows
    
    return df

def add_technical_indicators(df):
    # 1. Trend
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    
    # 2. Momentum
    df['RSI'] = ta.rsi(df['Close'], length=14)
    # Check if MACD returns a DataFrame and handle it
    macd = ta.macd(df['Close'])
    if isinstance(macd, pd.DataFrame):
        df['MACD'] = macd.iloc[:, 0] # fast line
    else:
        df['MACD'] = macd
    
    # 3. Volatility
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # 4. Bollinger Bands
    bb = ta.bbands(df['Close'], length=20)
    if bb is not None:
        df['BB_UPPER'] = bb.iloc[:, 0]
        df['BB_LOWER'] = bb.iloc[:, 2]

    # 5. SMC Features
    df = add_smc_indicators(df)

    df.dropna(inplace=True)
    return df

def preprocess_data(df):
    # REGRESSION TARGET: Predict the actual Close price of the next candle
    df['Target'] = df['Close'].shift(-FUTURE_TARGET)
    
    # We must drop the last row because it has no future target
    df.dropna(inplace=True)

    # Select features for the model
    # Added Dist_to_High, Dist_to_Low for SMC
    feature_cols = ['Close', 'RSI', 'MACD', 'ATR', 'EMA_50', 'BB_UPPER', 'BB_LOWER', 'Dist_to_High', 'Dist_to_Low']
    
    # Scale Features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])
    
    # Scale Target Separately (Crucial for Inverse Transform)
    target_scaler = MinMaxScaler()
    # Reshape target to (n_samples, 1) for scaler
    scaled_target = target_scaler.fit_transform(df[['Target']])
    
    X, y = [], []
    # Create sequences
    for i in range(SEQ_LEN, len(scaled_data)):
        X.append(scaled_data[i-SEQ_LEN:i])
        y.append(scaled_target[i]) # Use scaled target
        
    return np.array(X), np.array(y), scaler, target_scaler, feature_cols

# --- STEP 2: MODEL BUILDING FUNCTION ---

def build_model(input_shape):
    model = Sequential()
    
    # Layer 1: LSTM
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Layer 2: LSTM
    model.add(LSTM(units=128, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Layer 3: Dense
    model.add(Dense(64, activation='relu'))
    
    # Output Layer: LINEAR for Regression (Predicting a value, not a probability)
    model.add(Dense(1, activation='linear'))
    
    # Loss: Mean Squared Error for Regression
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    # 1. Load and Process
    df = load_data(CSV_FILE_PATH)
    df = add_technical_indicators(df)
    
    print("Preprocessing data sequences...")
    X, y, scaler, target_scaler, features = preprocess_data(df)

    # 2. Split Data (80% Train, 20% Test)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}")

    # 3. Save the Scalers (Crucial for the live bot)
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')
    print("Scalers saved as scaler.pkl and target_scaler.pkl")

    # 4. Build and Train Model
    print("Building Model...")
    model = build_model((X_train.shape[1], X_train.shape[2]))

    # Callbacks
    checkpoint = ModelCheckpoint("best_xauusd_model.keras", save_best_only=True, monitor='val_loss', mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Starting Training...")
    history = model.fit(
        X_train, y_train,
        epochs=20, 
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stop]
    )
    
    print("Training Complete. Model saved as best_xauusd_model.keras")