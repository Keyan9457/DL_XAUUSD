import time
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas_ta as ta
from live_trader import MT5RiskManager, get_trade_signal, ACCOUNT_BALANCE
from mt5_handler import initialize_mt5, get_mt5_data
from notification_handler import send_whatsapp_message
import datetime
import json
import os

# --- CONFIGURATION ---
MODEL_PATH = 'best_xauusd_model.keras'
SCALER_PATH = 'scaler.pkl'
TARGET_SCALER_PATH = 'target_scaler.pkl'
SYMBOL = 'XAUUSD' # MT5 Symbol (Check your broker, might be 'GOLD' or 'XAUUSD.m')
INTERVAL = '5m'
LOOKBACK_CANDLES = 300 
SEQ_LEN = 60 

def add_smc_indicators(df):
    """
    Adds simplified SMC features:
    1. Swing Highs/Lows (Fractals)
    2. Order Block Proximity (Distance to recent significant High/Low)
    """
    # Identify Swing Highs and Lows (Fractals) - Window of 5
    df['Swing_High'] = df['High'].rolling(window=5, center=True).max() == df['High']
    df['Swing_Low'] = df['Low'].rolling(window=5, center=True).min() == df['Low']
    
    # Forward fill the last known Swing High/Low
    df['Last_Swing_High'] = df['High'].where(df['Swing_High']).ffill()
    df['Last_Swing_Low'] = df['Low'].where(df['Swing_Low']).ffill()
    
    # Feature: Distance to Last Swing High/Low
    df['Dist_to_High'] = df['Last_Swing_High'] - df['Close']
    df['Dist_to_Low'] = df['Close'] - df['Last_Swing_Low']
    
    # Fill NaNs
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def calculate_indicators(df):
    # Copy from model_training.py to ensure consistency
    # 1. Trend
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    
    # 2. Momentum
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # Handle MACD
    macd = ta.macd(df['Close'])
    if isinstance(macd, pd.DataFrame):
        df['MACD'] = macd.iloc[:, 0]
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

def main():
    print("--- STARTING LIVE TRADER (MT5 + SMC + REGRESSION) ---")
    
    # 1. Initialize MT5
    if not initialize_mt5():
        print("Failed to connect to MT5. Exiting...")
        return

    # 2. Load Model and Scalers
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        target_scaler = joblib.load(TARGET_SCALER_PATH)
        print("Model and Scalers loaded successfully.")
    except Exception as e:
        print(f"Error loading model/scaler: {e}")
        return

    # 3. Initialize Manager
    manager = MT5RiskManager(current_balance=ACCOUNT_BALANCE, current_equity=ACCOUNT_BALANCE)
    
    # 4. Main Loop
    while True:
        try:
            print("\n--- New Cycle ---")
            
            # A. Check News
            if not manager.check_news():
                print("High Impact News Detected. Skipping trade.")
                time.sleep(300)
                time.sleep(300)
                continue
            
            # B. Check for Existing Trades (Prevent Multiple Executions)
            from mt5_handler import get_open_positions
            open_positions = get_open_positions(SYMBOL)
            if len(open_positions) > 0:
                print(f"⚠️ Position already open for {SYMBOL}. Skipping new analysis to prevent duplicate trades.")
                print(f"Open Position Ticket: {open_positions[0].ticket} | Profit: {open_positions[0].profit}")
                
                # Optional: Update Dashboard state even if we don't trade
                # For now, just wait
                time.sleep(60) 
                continue

            # C. Fetch Data (Multi-Timeframe)
            print(f"Fetching MTF data for {SYMBOL}...")
            df_30m = get_mt5_data(SYMBOL, "30m", 100)
            df_15m = get_mt5_data(SYMBOL, "15m", 100)
            df_5m = get_mt5_data(SYMBOL, "5m", LOOKBACK_CANDLES) # Main for AI
            df_3m = get_mt5_data(SYMBOL, "3m", 100)
            df_1m = get_mt5_data(SYMBOL, "1m", 100)
            
            if any(df is None for df in [df_30m, df_15m, df_5m, df_3m, df_1m]):
                print("Failed to get data for some timeframes. Retrying...")
                time.sleep(10)
                continue

            # --- FETCH CORRELATED ASSETS ---
            print("Fetching Correlated Assets (DXY, US10Y)...")
            
            # 1. DXY from MT5 (USDX)
            df_dxy = get_mt5_data("USDX", "5m", LOOKBACK_CANDLES)
            if df_dxy is not None:
                # Ensure indices are compatible (timezone naive)
                if df_5m.index.tz is not None:
                    df_5m.index = df_5m.index.tz_localize(None)
                if df_dxy.index.tz is not None:
                    df_dxy.index = df_dxy.index.tz_localize(None)
                    
                # Merge using reindex/ffill
                df_5m['DXY'] = df_dxy['Close'].reindex(df_5m.index, method='ffill')
            else:
                print("Warning: Could not fetch USDX from MT5. Using 0.0 (Risk of poor prediction)")
                df_5m['DXY'] = 0.0

            # 2. US10Y from YFinance (^TNX)
            try:
                import yfinance as yf
                # Fetch recent data to match 5m timeframe
                us10y_data = yf.download('^TNX', period='5d', interval='5m', progress=False)
                
                if not us10y_data.empty:
                    if us10y_data.index.tz is not None:
                        us10y_data.index = us10y_data.index.tz_localize(None)
                        
                    if isinstance(us10y_data.columns, pd.MultiIndex):
                        try:
                            us10y_close = us10y_data['Close']['^TNX']
                        except KeyError:
                            us10y_close = us10y_data['Close']
                    else:
                        us10y_close = us10y_data['Close']
                    
                    # Merge/Align with df_5m
                    aligned_us10y = us10y_close.reindex(df_5m.index, method='ffill')
                    df_5m['US10Y'] = aligned_us10y
                else:
                    print("Warning: No data for ^TNX")
                    df_5m['US10Y'] = 0.0
            except Exception as e:
                print(f"Error fetching US10Y: {e}")
                df_5m['US10Y'] = 0.0
            
            # Fill any remaining NaNs (e.g. if DXY/US10Y missing for latest candle)
            df_5m.fillna(method='ffill', inplace=True)
            df_5m.fillna(0, inplace=True)

            # C. Calculate Indicators (For Trend Checks)
            # Helper to get EMA
            def get_ema(df, length=50):
                return ta.ema(df['Close'], length=length).iloc[-1]
            
            # 1. HTF Trend (30m & 15m)
            ema_30m = get_ema(df_30m)
            close_30m = df_30m['Close'].iloc[-1]
            trend_30m = "BULLISH" if close_30m > ema_30m else "BEARISH"
            
            ema_15m = get_ema(df_15m)
            close_15m = df_15m['Close'].iloc[-1]
            trend_15m = "BULLISH" if close_15m > ema_15m else "BEARISH"
            
            htf_bias = "NEUTRAL"
            if trend_30m == "BULLISH" and trend_15m == "BULLISH":
                htf_bias = "BULLISH"
            elif trend_30m == "BEARISH" and trend_15m == "BEARISH":
                htf_bias = "BEARISH"
                
            print(f"HTF Bias (30m/15m): {htf_bias}")
            
            # 2. AI Prediction (5m)
            df_5m = calculate_indicators(df_5m)
            if len(df_5m) < SEQ_LEN:
                print("Not enough 5m data.")
                continue
                
            latest_5m = df_5m.tail(SEQ_LEN)
            feature_cols = ['Close', 'RSI', 'MACD', 'ATR', 'EMA_50', 'BB_UPPER', 'BB_LOWER', 'Dist_to_High', 'Dist_to_Low', 'DXY', 'US10Y']
            signal_ai, predicted_price, confidence = get_trade_signal(latest_5m, model, scaler, target_scaler, feature_cols)
            
            current_price = latest_5m['Close'].iloc[-1]
            
            # 3. LTF Confirmation (3m & 1m)
            ema_3m = get_ema(df_3m)
            close_3m = df_3m['Close'].iloc[-1]
            trend_3m = "BULLISH" if close_3m > ema_3m else "BEARISH"
            
            ema_1m = get_ema(df_1m)
            close_1m = df_1m['Close'].iloc[-1]
            trend_1m = "BULLISH" if close_1m > ema_1m else "BEARISH"
            
            ltf_conf = "NEUTRAL"
            if trend_3m == "BULLISH" and trend_1m == "BULLISH":
                ltf_conf = "BULLISH"
            elif trend_3m == "BEARISH" and trend_1m == "BEARISH":
                ltf_conf = "BEARISH"
                
            print(f"AI Signal (5m): {signal_ai}")
            print(f"LTF Conf (3m/1m): {ltf_conf}")
            
            # --- FINAL DECISION (WATERFALL) ---
            final_signal = "HOLD"
            
            if htf_bias == "BULLISH" and signal_ai == "BUY" and ltf_conf == "BULLISH":
                final_signal = "BUY"
            elif htf_bias == "BEARISH" and signal_ai == "SELL" and ltf_conf == "BEARISH":
                final_signal = "SELL"
            else:
                print("Mismatch in Timeframes. Waiting for perfect setup...")
            
            # Time Output
            last_time_naive = latest_5m.index[-1]
            last_time_eet = last_time_naive.tz_localize('Europe/Helsinki')
            last_time_ist = last_time_eet.tz_convert('Asia/Kolkata')
            
            print(f"Candle Time: {last_time_ist.strftime('%Y-%m-%d %I:%M:%S %p')} (Chennai)")
            print(f"Current Price: {current_price:.2f}")
            print(f"Predicted Price: {predicted_price:.2f}")
            print(f"Candle Time: {last_time_ist.strftime('%Y-%m-%d %I:%M:%S %p')} (Chennai)")
            print(f"Current Price: {current_price:.2f}")
            print(f"Predicted Price: {predicted_price:.2f}")
            print(f"Confidence: {confidence:.1f}%")
            print(f"Final Signal: {final_signal}")
            
            # --- SMC STOP LOSS LOGIC (Always Calculate for Display) ---
            last_swing_low = latest_5m['Last_Swing_Low'].iloc[-1]
            last_swing_high = latest_5m['Last_Swing_High'].iloc[-1]
            atr = latest_5m['ATR'].iloc[-1]
            
            # Calculate Potential SL/TP for both scenarios to show user
            pot_buy_sl = last_swing_low - 0.50 if (last_swing_low > 0 and last_swing_low < current_price) else current_price - (atr * 1.5)
            pot_sell_sl = last_swing_high + 0.50 if (last_swing_high > 0 and last_swing_high > current_price) else current_price + (atr * 1.5)
            
            # Buy Metrics
            buy_risk = current_price - pot_buy_sl
            buy_tp = current_price + (buy_risk * 2)
            buy_risk_pct = (buy_risk / current_price) * 100
            buy_reward_pct = (buy_risk * 2 / current_price) * 100
            
            # Sell Metrics
            sell_risk = pot_sell_sl - current_price
            sell_tp = current_price - (sell_risk * 2)
            sell_risk_pct = (sell_risk / current_price) * 100
            sell_reward_pct = (sell_risk * 2 / current_price) * 100
            
            print(f"Potential BUY:  Entry {current_price:.2f} | SL {pot_buy_sl:.2f} (-{buy_risk_pct:.2f}%) | TP {buy_tp:.2f} (+{buy_reward_pct:.2f}%) | R:R 1:2")
            print(f"Potential SELL: Entry {current_price:.2f} | SL {pot_sell_sl:.2f} (-{sell_risk_pct:.2f}%) | TP {sell_tp:.2f} (+{sell_reward_pct:.2f}%) | R:R 1:2")

            if final_signal == "BUY" or final_signal == "SELL":
                print("Est. Duration: 5-15 Mins (Scalp)")
                
                stop_loss = 0.0
                take_profit = 0.0
                
                if final_signal == "BUY":
                    stop_loss = pot_buy_sl
                    risk = current_price - stop_loss
                    take_profit = current_price + (risk * 2)
                    
                elif final_signal == "SELL":
                    stop_loss = pot_sell_sl
                    risk = stop_loss - current_price
                    take_profit = current_price - (risk * 2)

                print(f"Stop Loss: {stop_loss:.2f}")
                print(f"Take Profit: {take_profit:.2f}")
                    
                lots = manager.calculate_position_size(current_price, stop_loss)
                


                if manager.check_drawdown_rules(0): 
                    manager.execute_trade(final_signal, lots, current_price, stop_loss, take_profit, SYMBOL)
                    
                    # Send WhatsApp Alert
                    msg = f"🚨 *TRADE EXECUTED* 🚨\n\n" \
                          f"Symbol: {SYMBOL}\n" \
                          f"Type: {final_signal}\n" \
                          f"Entry: {current_price:.2f}\n" \
                          f"SL: {stop_loss:.2f}\n" \
                          f"TP: {take_profit:.2f}\n" \
                          f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}"
                    
                    send_whatsapp_message(msg)
                else:
                    print("Trade rejected by Risk Manager (Drawdown Limit).")
            
            # --- SAVE STATE FOR DASHBOARD ---
            # Prepare Candle Data for Chart (Last 50 candles)
            chart_data = []
            if not latest_5m.empty:
                # Ensure index is datetime and reset it to get it as a column
                temp_df = latest_5m.tail(50).copy()
                temp_df.reset_index(inplace=True)
                # Convert to list of dicts: [{'Date': '...', 'Open': ...}, ...]
                # Note: JSON serialization of Timestamps needs string conversion
                temp_df['Date'] = temp_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                chart_data = temp_df[['Date', 'Open', 'High', 'Low', 'Close']].to_dict('records')

            state = {
                "last_update": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "symbol": SYMBOL,
                "current_price": float(current_price),
                "predicted_price": float(predicted_price),
                "signals": {
                    "htf_bias": htf_bias,
                    "ai_signal": signal_ai,
                    "confidence": confidence,
                    "ltf_conf": ltf_conf,
                    "final_signal": final_signal
                },
                "trade_setup": {
                    "buy": {
                        "entry": float(current_price),
                        "sl": float(pot_buy_sl),
                        "tp": float(buy_tp)
                    },
                    "sell": {
                        "entry": float(current_price),
                        "sl": float(pot_sell_sl),
                        "tp": float(sell_tp)
                    }
                },
                "chart_data": chart_data
            }
            
            try:
                with open('bot_state.json', 'w') as f:
                    json.dump(state, f, indent=4)
                print("Dashboard state updated.")
            except Exception as e:
                print(f"Error saving dashboard state: {e}")

            print("Bot is Active. Waiting for next cycle...")
            for i in range(300, 0, -1):
                print(f"Next cycle in {i} seconds...", end='\r')
                time.sleep(1)
            print(" " * 30, end='\r') # Clear line 
            
        except KeyboardInterrupt:
            print("\nStopping Bot...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
