import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from mt5_handler import initialize_mt5, get_mt5_data

SYMBOL = 'XAUUSD'
LOOKBACK_CANDLES = 300

# Load scaler to check expected features
scaler = joblib.load('scaler.pkl')
print(f"Scaler expects {scaler.n_features_in_} features")

# Initialize MT5
if not initialize_mt5():
    print("Failed to connect to MT5")
    exit()

# Fetch data
print("Fetching data...")
df_15m = get_mt5_data(SYMBOL, "15m", 200)
df_1h = get_mt5_data(SYMBOL, "1h", 200)
df_4h = get_mt5_data(SYMBOL, "4h", 200)
df_5m = get_mt5_data(SYMBOL, "5m", LOOKBACK_CANDLES)

print(f"df_5m shape: {df_5m.shape}")
print(f"df_15m shape: {df_15m.shape}")
print(f"df_1h shape: {df_1h.shape}")
print(f"df_4h shape: {df_4h.shape}")

# Test adding one HTF feature to see all columns
def add_smc_indicators(df):
    df['Swing_High'] = df['High'].rolling(window=5, center=True).max() == df['High']
    df['Swing_Low'] = df['Low'].rolling(window=5, center=True).min() == df['Low']
    df['Last_Swing_High'] = df['High'].where(df['Swing_High']).ffill()
    df['Last_Swing_Low'] = df['Low'].where(df['Swing_Low']).ffill()
    df['Dist_to_High'] = df['Last_Swing_High'] - df['Close']
    df['Dist_to_Low'] = df['Close'] - df['Last_Swing_Low']
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    return df

df_test = df_15m.copy()
df_test['EMA_50'] = ta.ema(df_test['Close'], length=50)
df_test['EMA_200'] = ta.ema(df_test['Close'], length=200)
df_test['RSI'] = ta.rsi(df_test['Close'], length=14)
macd = ta.macd(df_test['Close'])
if isinstance(macd, pd.DataFrame):
    df_test['MACD'] = macd.iloc[:, 0]
else:
    df_test['MACD'] = macd
df_test['ATR'] = ta.atr(df_test['High'], df_test['Low'], df_test['Close'], length=14)
bb = ta.bbands(df_test['Close'], length=20)
if bb is not None:
    df_test['BB_UPPER'] = bb.iloc[:, 0]
    df_test['BB_LOWER'] = bb.iloc[:, 2]
df_test = add_smc_indicators(df_test)
if 'Volume' in df_test.columns:
    df_test['Volume_MA'] = df_test['Volume'].rolling(window=20).mean()
    df_test['Volume_Ratio'] = df_test['Volume'] / df_test['Volume_MA']
df_test['Log_Return'] = np.log(df_test['Close'] / df_test['Close'].shift(1))
df_test.dropna(inplace=True)

htf_cols = [col for col in df_test.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"\nHTF15 columns after indicators ({len(htf_cols)}):")
for col in sorted(htf_cols):
    print(f"  - HTF15_{col}")
