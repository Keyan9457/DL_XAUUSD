"""
Backtest SMC Trading System
Evaluates the 4-gate system (HTF + AI + LTF + SMC) on historical data
"""

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas_ta as ta
from mt5_handler import initialize_mt5, get_mt5_data
from smc_indicators import add_all_smc_indicators, get_smc_signal
import MetaTrader5 as mt5
from datetime import datetime, timedelta

# Configuration
MODEL_PATH = 'best_xauusd_model.keras'
SCALER_PATH = 'scaler.pkl'
TARGET_SCALER_PATH = 'target_scaler.pkl'
SYMBOL = 'XAUUSD'
SEQ_LEN = 60
RISK_PERCENT = 0.02
INITIAL_BALANCE = 5000

def calculate_indicators(df, prefix=''):
    """Calculate technical indicators"""
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
        df = add_all_smc_indicators(df)
    
    if 'Volume' in df.columns:
        df[f'{prefix}Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df[f'{prefix}Volume_Ratio'] = df['Volume'] / df[f'{prefix}Volume_MA']
    
    df.dropna(inplace=True)
    return df

def get_htf_bias(df_4h):
    """Get higher timeframe bias"""
    if len(df_4h) < 200:
        return "NEUTRAL"
    
    latest = df_4h.iloc[-1]
    ema_50 = latest['EMA_50']
    ema_200 = latest['EMA_200']
    
    if ema_50 > ema_200:
        return "BULLISH"
    elif ema_50 < ema_200:
        return "BEARISH"
    return "NEUTRAL"

def get_ltf_confirmation(df_3m, df_1m):
    """Get lower timeframe confirmation"""
    if len(df_3m) < 50 or len(df_1m) < 50:
        return "NEUTRAL"
    
    ema_3m = df_3m['EMA_50'].iloc[-1]
    close_3m = df_3m['Close'].iloc[-1]
    trend_3m = "BULLISH" if close_3m > ema_3m else "BEARISH"
    
    ema_1m = df_1m['EMA_50'].iloc[-1]
    close_1m = df_1m['Close'].iloc[-1]
    trend_1m = "BULLISH" if close_1m > ema_1m else "BEARISH"
    
    if trend_3m == "BULLISH" and trend_1m == "BULLISH":
        return "BULLISH"
    elif trend_3m == "BEARISH" and trend_1m == "BEARISH":
        return "BEARISH"
    return "NEUTRAL"

def backtest_smc_system(days=30):
    """
    Backtest the 4-gate SMC system
    Returns: DataFrame with trade results and performance metrics
    """
    print("="*60)
    print("SMC TRADING SYSTEM BACKTEST")
    print("="*60)
    print(f"Testing Period: Last {days} days")
    print(f"Initial Balance: ${INITIAL_BALANCE}")
    print(f"Risk per Trade: {RISK_PERCENT*100}%")
    print("="*60)
    
    # Initialize MT5
    if not initialize_mt5():
        print("Failed to initialize MT5")
        return None
    
    # Load model and scalers
    print("\nLoading AI model...")
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    
    # Fetch historical data
    print("Fetching historical data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 10)  # Extra days for indicators
    
    df_5m = get_mt5_data(SYMBOL, '5m', count=days*288)  # 288 5m candles per day
    df_4h = get_mt5_data(SYMBOL, '4h', count=days*6 + 50)
    df_3m = get_mt5_data(SYMBOL, '3m', count=days*480)
    df_1m = get_mt5_data(SYMBOL, '1m', count=days*1440)
    
    if df_5m is None or df_4h is None:
        print("Failed to fetch data")
        return None
    
    # Calculate indicators
    print("Calculating indicators...")
    df_5m = calculate_indicators(df_5m)
    df_4h = calculate_indicators(df_4h, prefix='HTF_')
    if df_3m is not None:
        df_3m = calculate_indicators(df_3m, prefix='LTF3_')
    if df_1m is not None:
        df_1m = calculate_indicators(df_1m, prefix='LTF1_')
    
    # Prepare features
    feature_cols = [col for col in df_5m.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Backtest
    trades = []
    balance = INITIAL_BALANCE
    
    print("\nRunning backtest...")
    print("-"*60)
    
    for i in range(SEQ_LEN + 200, len(df_5m) - 1):
        current_time = df_5m.index[i]
        current_price = df_5m['Close'].iloc[i]
        
        # Get HTF bias
        htf_idx = df_4h.index.get_indexer([current_time], method='ffill')[0]
        if htf_idx < 0 or htf_idx >= len(df_4h):
            continue
        htf_bias = get_htf_bias(df_4h.iloc[:htf_idx+1])
        
        # Get AI signal
        sequence = df_5m[feature_cols].iloc[i-SEQ_LEN:i].values
        sequence_scaled = scaler.transform(sequence)
        sequence_reshaped = sequence_scaled.reshape(1, SEQ_LEN, -1)
        
        prediction_scaled = model.predict(sequence_reshaped, verbose=0)
        prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]
        predicted_price = current_price * np.exp(prediction)
        
        confidence = abs(predicted_price - current_price) / current_price
        signal_ai = "BUY" if predicted_price > current_price else "SELL"
        
        # Get LTF confirmation
        ltf_conf = "NEUTRAL"
        if df_3m is not None and df_1m is not None:
            ltf3_idx = df_3m.index.get_indexer([current_time], method='ffill')[0]
            ltf1_idx = df_1m.index.get_indexer([current_time], method='ffill')[0]
            if ltf3_idx >= 0 and ltf1_idx >= 0:
                ltf_conf = get_ltf_confirmation(
                    df_3m.iloc[:ltf3_idx+1],
                    df_1m.iloc[:ltf1_idx+1]
                )
        
        # Get SMC signal
        smc_signal, smc_confidence, smc_reason = get_smc_signal(
            df_5m.iloc[:i+1],
            current_price
        )
        
        # 4-Gate Decision
        final_signal = "HOLD"
        
        if (htf_bias == "BULLISH" and 
            signal_ai == "BUY" and 
            ltf_conf == "BULLISH" and 
            smc_signal == "BUY" and 
            smc_confidence >= 0.3):
            final_signal = "BUY"
        elif (htf_bias == "BEARISH" and 
              signal_ai == "SELL" and 
              ltf_conf == "BEARISH" and 
              smc_signal == "SELL" and 
              smc_confidence >= 0.3):
            final_signal = "SELL"
        
        # Execute trade
        if final_signal in ["BUY", "SELL"]:
            atr = df_5m['ATR'].iloc[i]
            
            # Calculate SL and TP
            if final_signal == "BUY":
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 4)
            else:
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 4)
            
            # Calculate position size
            risk_amount = balance * RISK_PERCENT
            sl_distance = abs(current_price - stop_loss)
            lots = risk_amount / sl_distance
            
            # Simulate trade outcome
            future_prices = df_5m['Close'].iloc[i+1:i+100]  # Next 100 candles
            
            if len(future_prices) == 0:
                continue
            
            hit_tp = False
            hit_sl = False
            exit_price = future_prices.iloc[-1]
            
            for future_price in future_prices:
                if final_signal == "BUY":
                    if future_price >= take_profit:
                        hit_tp = True
                        exit_price = take_profit
                        break
                    elif future_price <= stop_loss:
                        hit_sl = True
                        exit_price = stop_loss
                        break
                else:  # SELL
                    if future_price <= take_profit:
                        hit_tp = True
                        exit_price = take_profit
                        break
                    elif future_price >= stop_loss:
                        hit_sl = True
                        exit_price = stop_loss
                        break
            
            # Calculate P&L
            if final_signal == "BUY":
                pnl = (exit_price - current_price) * lots
            else:
                pnl = (current_price - exit_price) * lots
            
            balance += pnl
            
            # Record trade
            trades.append({
                'time': current_time,
                'signal': final_signal,
                'entry': current_price,
                'exit': exit_price,
                'sl': stop_loss,
                'tp': take_profit,
                'lots': lots,
                'pnl': pnl,
                'balance': balance,
                'hit_tp': hit_tp,
                'hit_sl': hit_sl,
                'htf_bias': htf_bias,
                'ai_signal': signal_ai,
                'ltf_conf': ltf_conf,
                'smc_signal': smc_signal,
                'smc_confidence': smc_confidence,
                'smc_reason': smc_reason
            })
            
            print(f"{current_time} | {final_signal} | Entry: {current_price:.2f} | "
                  f"Exit: {exit_price:.2f} | P&L: ${pnl:.2f} | Balance: ${balance:.2f}")
    
    # Calculate statistics
    if len(trades) == 0:
        print("\n⚠️ No trades generated during backtest period")
        return None
    
    df_trades = pd.DataFrame(trades)
    
    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades['pnl'] > 0])
    losing_trades = len(df_trades[df_trades['pnl'] < 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    total_pnl = df_trades['pnl'].sum()
    avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
    
    profit_factor = abs(df_trades[df_trades['pnl'] > 0]['pnl'].sum() / 
                       df_trades[df_trades['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
    
    max_balance = df_trades['balance'].max()
    min_balance = df_trades['balance'].min()
    max_drawdown = ((max_balance - min_balance) / max_balance) * 100
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"\nAverage Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"\nTotal P&L: ${total_pnl:.2f}")
    print(f"Final Balance: ${balance:.2f}")
    print(f"Return: {((balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100):.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print("="*60)
    
    # Save results
    df_trades.to_csv('backtest_results.csv', index=False)
    print("\n✅ Results saved to backtest_results.csv")
    
    return df_trades

if __name__ == "__main__":
    # Run backtest
    results = backtest_smc_system(days=30)
    
    if results is not None:
        print("\n📊 Backtest completed successfully!")
        print("Check backtest_results.csv for detailed trade log")
