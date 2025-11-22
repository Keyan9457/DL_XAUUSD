import yfinance as yf
import pandas as pd
import pandas_ta as ta

print("Downloading data...")
df = yf.download('GC=F', period='5d', interval='5m')
print("Data shape:", df.shape)
print("Columns:", df.columns)
print("Head:\n", df.head())

print("\nChecking Close column...")
try:
    close = df['Close']
    print("Close type:", type(close))
    print("Close head:\n", close.head())
except Exception as e:
    print("Error accessing Close:", e)

print("\nChecking MACD...")
try:
    macd = ta.macd(df['Close'])
    print("MACD type:", type(macd))
    print("MACD head:\n", macd.head())
except Exception as e:
    print("Error calculating MACD:", e)
