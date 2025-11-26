import time
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas_ta as ta
from live_trader import MT5RiskManager, get_trade_signal, ACCOUNT_BALANCE
from mt5_handler import initialize_mt5, get_mt5_data
from notification_handler import send_whatsapp_message
from smc_indicators import add_all_smc_indicators, get_smc_signal
import datetime
import json
import os
import threading
import MetaTrader5 as mt5

# --- CONFIGURATION ---
MODEL_PATH = 'best_xauusd_model.keras'
SCALER_PATH = 'scaler.pkl'
TARGET_SCALER_PATH = 'target_scaler.pkl'
SYMBOL = 'XAUUSD' # MT5 Symbol (Check your broker, might be 'GOLD' or 'XAUUSD.m')
INTERVAL = '5m'
LOOKBACK_CANDLES = 80  # Reduced for demo account compatibility
SEQ_LEN = 60 

# SMC indicators are now handled by smc_indicators.py module
# See add_all_smc_indicators() for full implementation


def calculate_indicators(df, prefix=''):
    # Copy from model_training.py to ensure consistency
    # 1. Trend
    df[f'{prefix}EMA_50'] = ta.ema(df['Close'], length=50)
    df[f'{prefix}EMA_200'] = ta.ema(df['Close'], length=200)
    
    # 2. Momentum
    df[f'{prefix}RSI'] = ta.rsi(df['Close'], length=14)
    
    # Handle MACD
    macd = ta.macd(df['Close'])
    if isinstance(macd, pd.DataFrame):
        df[f'{prefix}MACD'] = macd.iloc[:, 0]
    else:
        df[f'{prefix}MACD'] = macd
    
    # 3. Volatility
    df[f'{prefix}ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # 4. Bollinger Bands
    bb = ta.bbands(df['Close'], length=20)
    if bb is not None:
        df[f'{prefix}BB_UPPER'] = bb.iloc[:, 0]
        df[f'{prefix}BB_LOWER'] = bb.iloc[:, 2]

    # 5. SMC Features (only for main timeframe)
    if prefix == '':
        df = add_all_smc_indicators(df)
    
    # 6. Volume Features
    if 'Volume' in df.columns:
        df[f'{prefix}Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df[f'{prefix}Volume_Ratio'] = df['Volume'] / df[f'{prefix}Volume_MA']

    df.dropna(inplace=True)
    return df

def align_htf_features(df_main, df_htf, prefix):
    """Align higher timeframe features with main dataframe (matching training process)"""
    # Calculate indicators for HTF (including SMC for HTF)
    df_htf_copy = df_htf.copy()
    
    # Add all technical indicators (this includes SMC features for HTF too)
    df_htf_copy[f'EMA_50'] = ta.ema(df_htf_copy['Close'], length=50)
    df_htf_copy[f'EMA_200'] = ta.ema(df_htf_copy['Close'], length=200)
    df_htf_copy[f'RSI'] = ta.rsi(df_htf_copy['Close'], length=14)
    
    macd = ta.macd(df_htf_copy['Close'])
    if isinstance(macd, pd.DataFrame):
        df_htf_copy[f'MACD'] = macd.iloc[:, 0]
    else:
        df_htf_copy[f'MACD'] = macd
    
    df_htf_copy[f'ATR'] = ta.atr(df_htf_copy['High'], df_htf_copy['Low'], df_htf_copy['Close'], length=14)
    
    bb = ta.bbands(df_htf_copy['Close'], length=20)
    if bb is not None:
        df_htf_copy[f'BB_UPPER'] = bb.iloc[:, 0]
        df_htf_copy[f'BB_LOWER'] = bb.iloc[:, 2]
    
    # SMC Features for HTF
    df_htf_copy = add_all_smc_indicators(df_htf_copy)
    
    # Volume Features
    if 'Volume' in df_htf_copy.columns:
        df_htf_copy[f'Volume_MA'] = df_htf_copy['Volume'].rolling(window=20).mean()
        df_htf_copy[f'Volume_Ratio'] = df_htf_copy['Volume'] / df_htf_copy[f'Volume_MA']
    
    # Log Return
    df_htf_copy[f'Log_Return'] = np.log(df_htf_copy['Close'] / df_htf_copy['Close'].shift(1))
    
    df_htf_copy.dropna(inplace=True)
    
    # Select only the indicator columns (exclude OHLCV)
    htf_cols = [col for col in df_htf_copy.columns 
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    df_htf_selected = df_htf_copy[htf_cols].copy()
    df_htf_selected.columns = [f'{prefix}{col}' for col in df_htf_selected.columns]
    
    # Ensure indices are timezone-naive for alignment
    if df_main.index.tz is not None:
        df_main.index = df_main.index.tz_localize(None)
    if df_htf_selected.index.tz is not None:
        df_htf_selected.index = df_htf_selected.index.tz_localize(None)
    
    # Join and forward-fill
    df_merged = df_main.join(df_htf_selected, how='left')
    df_merged.fillna(method='ffill', inplace=True)
    
    return df_merged

# Global flag to control the price update thread
price_update_running = True
file_lock = threading.Lock()

def update_live_price():
    """Background thread to update live price every 2 seconds"""
    global price_update_running
    while price_update_running:
        try:
            # Get current tick price from MT5
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is not None:
                current_price = tick.bid
                
                # Update bot_state.json with live price only
                try:
                    # Read existing state
                    if os.path.exists('bot_state.json'):
                        with open('bot_state.json', 'r') as f:
                            state = json.load(f)
                        
                        # Update only the price and timestamp
                        state['current_price'] = float(current_price)
                        state['last_price_update'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Write back
                        with file_lock:
                            with open('bot_state.json', 'w') as f:
                                json.dump(state, f, indent=4)
                except Exception as e:
                    pass  # Silently skip errors to not interrupt the thread
            
            time.sleep(0.5)  # Update every 0.5 seconds for near-real-time ticks
        except Exception as e:
            time.sleep(0.5)

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
    
    # 4. Start Live Price Update Thread
    print("Starting live price update thread...")
    price_thread = threading.Thread(target=update_live_price, daemon=True)
    price_thread.start()
    
    # 5. Main Loop
    while True:
        try:
            print("\n--- New Cycle ---")
            
            # A. Check News
            # A. Check News
            if not manager.check_news():
                print("High Impact News Detected. Skipping trade.")
                
                # Save "Waiting" state for dashboard
                try:
                    state = {
                        "last_update": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "symbol": SYMBOL,
                        "current_price": 0.0, # Will be updated by thread
                        "predicted_price": 0.0,
                        "next_update": (datetime.datetime.now() + datetime.timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S'),
                        "signals": {
                            "htf_bias": "NEWS",
                            "ai_signal": "WAIT",
                            "confidence": 0.0,
                            "ltf_conf": "NEWS",
                            "smc_signal": "WAIT",
                            "smc_confidence": 0.0,
                            "smc_reason": "High Impact News Event",
                            "final_signal": "HOLD"
                        },
                        "trade_setup": {}
                    }
                    with open('bot_state.json', 'w') as f:
                        json.dump(state, f, indent=4)
                except Exception as e:
                    print(f"Error saving state: {e}")

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
            df_15m = get_mt5_data(SYMBOL, "15m", 500)  # Increased for EMA 200
            df_1h = get_mt5_data(SYMBOL, "1h", 500)    # Increased for EMA 200
            df_4h = get_mt5_data(SYMBOL, "4h", 500)    # Increased for EMA 200
            df_5m = get_mt5_data(SYMBOL, "5m", LOOKBACK_CANDLES + 100) # Main for AI
            df_3m = get_mt5_data(SYMBOL, "3m", 100)
            df_1m = get_mt5_data(SYMBOL, "1m", 100)
            
            if any(df is None for df in [df_30m, df_15m, df_1h, df_4h, df_5m, df_3m, df_1m]):
                print("Failed to get data for some timeframes. Retrying...")
                time.sleep(10)
                continue

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
            
            # 2. AI Prediction (5m) - Calculate indicators and align HTF features
            # NOTE: DXY and US10Y were NOT in training data, removed for now
            # To use them, retrain the model with these features included
            
            df_5m = calculate_indicators(df_5m, prefix='')
            print("DEBUG: df_5m columns:", df_5m.columns.tolist())
            if 'Dist_to_High' not in df_5m.columns:
                print("CRITICAL: Dist_to_High MISSING in df_5m")
            else:
                print("OK: Dist_to_High present in df_5m")
            
            # Align HTF features (matching training process)
            print("Aligning HTF features...")
            df_5m = align_htf_features(df_5m, df_15m, 'HTF15_')
            df_5m = align_htf_features(df_5m, df_1h, 'HTF1H_')
            df_5m = align_htf_features(df_5m, df_4h, 'HTF4H_')
            
            if len(df_5m) < SEQ_LEN:
                print("Not enough 5m data.")
                continue
                
            latest_5m = df_5m.tail(SEQ_LEN)
            
            # Feature columns MUST match training order EXACTLY (from scaler)
            feature_cols = [
                # Base features (M5) - 12 features
                'Close', 'RSI', 'MACD', 'ATR', 'EMA_50', 'EMA_200', 'BB_UPPER', 'BB_LOWER',
                'Dist_to_High', 'Dist_to_Low', 'Volume_MA', 'Volume_Ratio',
                # HTF15 features - 16 features (exact scaler order)
                'HTF15_EMA_50', 'HTF15_EMA_200', 'HTF15_RSI', 'HTF15_MACD', 'HTF15_ATR',
                'HTF15_BB_UPPER', 'HTF15_BB_LOWER', 'HTF15_Swing_High', 'HTF15_Swing_Low',
                'HTF15_Last_Swing_High', 'HTF15_Last_Swing_Low', 'HTF15_Dist_to_High', 'HTF15_Dist_to_Low',
                'HTF15_Volume_MA', 'HTF15_Volume_Ratio', 'HTF15_Log_Return',
                # HTF1H features - 16 features (exact scaler order)
                'HTF1H_EMA_50', 'HTF1H_EMA_200', 'HTF1H_RSI', 'HTF1H_MACD', 'HTF1H_ATR',
                'HTF1H_BB_UPPER', 'HTF1H_BB_LOWER', 'HTF1H_Swing_High', 'HTF1H_Swing_Low',
                'HTF1H_Last_Swing_High', 'HTF1H_Last_Swing_Low', 'HTF1H_Dist_to_High', 'HTF1H_Dist_to_Low',
                'HTF1H_Volume_MA', 'HTF1H_Volume_Ratio', 'HTF1H_Log_Return',
                # HTF4H features - 16 features (exact scaler order)
                'HTF4H_EMA_50', 'HTF4H_EMA_200', 'HTF4H_RSI', 'HTF4H_MACD', 'HTF4H_ATR',
                'HTF4H_BB_UPPER', 'HTF4H_BB_LOWER', 'HTF4H_Swing_High', 'HTF4H_Swing_Low',
                'HTF4H_Last_Swing_High', 'HTF4H_Last_Swing_Low', 'HTF4H_Dist_to_High', 'HTF4H_Dist_to_Low',
                'HTF4H_Volume_MA', 'HTF4H_Volume_Ratio', 'HTF4H_Log_Return'
            ]
            # Total: 12 + 16 + 16 + 16 = 60 features
            
            # Filter to only columns that exist (defensive programming)
            feature_cols = [col for col in feature_cols if col in latest_5m.columns]
            
            print(f"Using {len(feature_cols)} features for prediction (expected: 60)")
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
            
            # --- SMC SIGNAL (4th Gate) ---
            smc_signal, smc_confidence, smc_reason = get_smc_signal(latest_5m, current_price)
            print(f"SMC Signal: {smc_signal} (Confidence: {smc_confidence:.2f})")
            print(f"SMC Reason: {smc_reason}")
            
            # --- FINAL DECISION (4-GATE SYSTEM) ---
            final_signal = "HOLD"
            
            # Gate 1: HTF Bias
            # Gate 2: AI Model Signal
            # Gate 3: LTF Confirmation
            # Gate 4: SMC Confirmation (NEW!)
            
            if (htf_bias == "BULLISH" and 
                signal_ai == "BUY" and 
                ltf_conf == "BULLISH" and 
                smc_signal == "BUY" and 
                smc_confidence >= 0.3):  # Minimum 30% SMC confidence
                final_signal = "BUY"
                print("✅ ALL 4 GATES ALIGNED - BUY SIGNAL CONFIRMED!")
                
            elif (htf_bias == "BEARISH" and 
                  signal_ai == "SELL" and 
                  ltf_conf == "BEARISH" and 
                  smc_signal == "SELL" and 
                  smc_confidence >= 0.3):
                final_signal = "SELL"
                print("✅ ALL 4 GATES ALIGNED - SELL SIGNAL CONFIRMED!")
            else:
                print("⚠️ Gates not aligned. Waiting for perfect setup...")
                if smc_signal == "HOLD":
                    print("   → No clear SMC setup detected")
            
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
                    "smc_signal": smc_signal,
                    "smc_confidence": smc_confidence,
                    "smc_reason": smc_reason,
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
                "chart_data": chart_data,
                "next_update": (datetime.datetime.now() + datetime.timedelta(seconds=300)).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            try:
                with file_lock:
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
