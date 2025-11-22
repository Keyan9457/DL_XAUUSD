import numpy as np
import joblib
from tensorflow.keras.models import load_model
import feedparser # For news
import datetime

# --- CONSTANTS ---
ACCOUNT_BALANCE = 5000.0
LEVERAGE = 20
MAX_DAILY_LOSS_PCT = 0.03 # 3%
MAX_RISK_PER_TRADE_PCT = 0.01 # 1% ($50)
PIP_VALUE_XAUUSD = 10 # Approx $10 per lot per pip (varies by broker)
CONSISTENCY_LIMIT = 0.15 # 15%

class MT5RiskManager:
    def __init__(self, current_balance, current_equity):
        self.balance = current_balance
        self.equity = current_equity
        self.daily_start_equity = current_balance # Simplification for example
        
    def check_news(self):
        """
        Checks ForexFactory RSS feed for high impact USD or XAU news.
        Returns True if SAFE to trade, False if NEWS imminent.
        """
        from news_handler import is_trading_safe
        safe, reason = is_trading_safe()
        if not safe:
            print(f"NEWS FILTER: {reason}")
        return safe

    def check_drawdown_rules(self, current_loss):
        # Rule 2.1: Max Daily Loss 3%
        daily_loss = self.daily_start_equity - self.equity
        if daily_loss + current_loss >= (self.daily_start_equity * MAX_DAILY_LOSS_PCT):
            return False # Trade denied: Would hit daily limit
        return True

    def calculate_position_size(self, entry_price, stop_loss_price):
        """
        Calculates lot size based on Risk ($50) and Leverage Constraints.
        """
        risk_amount = self.balance * MAX_RISK_PER_TRADE_PCT # $50
        
        # Calculate distance in pips
        sl_distance = abs(entry_price - stop_loss_price)
        # Gold pip: 2000.10 -> 2000.20 is 1 pip usually (0.10 movement)
        # Adjust logic based on your broker's digit precision
        pips = sl_distance * 10 
        
        if pips == 0: return 0
        
        # Formula: Lots = Risk / (Pips * PipValue)
        lot_size = risk_amount / (pips * PIP_VALUE_XAUUSD)
        
        # Leverage Check: Max Notional = $5000 * 20 = $100,000
        # 1 Lot Gold ~ $265,000 (at $2650 price)
        # Max Lots allowed approx 0.37
        max_lots_leverage = (self.balance * LEVERAGE) / (entry_price * 100) 
        
        final_lots = min(lot_size, max_lots_leverage)
        return round(final_lots, 2)

    def execute_trade(self, signal, lots, price, sl, tp, symbol):
        """
        Executes trade via MT5 Handler.
        """
        from mt5_handler import place_mt5_order
        
        print(f"--- EXECUTING TRADE ---")
        print(f"Symbol: {symbol}")
        print(f"Type: {signal}")
        print(f"Lots: {lots}")
        print(f"Price: {price}")
        print(f"SL: {sl} | TP: {tp}")
        
        if place_mt5_order(symbol, signal, lots, sl, tp):
            print("Trade executed successfully in MT5.")
        else:
            print("Trade execution failed in MT5.")
        print(f"----------------------")

# --- LIVE PREDICTION FUNCTION ---
def get_trade_signal(latest_60_candles_df, model, scaler, target_scaler, features):
    # 1. Preprocess Data
    data = latest_60_candles_df[features].values
    scaled_data = scaler.transform(data)
    X_input = np.array([scaled_data]) # Reshape to (1, 60, features)
    
    # 2. Predict (Returns scaled price)
    scaled_prediction = model.predict(X_input, verbose=0)[0][0]
    
    # 3. Inverse Transform to get Real Price
    predicted_price = target_scaler.inverse_transform([[scaled_prediction]])[0][0]
    
    # 4. Signal Logic
    current_price = latest_60_candles_df['Close'].iloc[-1]
    
    # Threshold: Only trade if predicted move is > $1.00 (approx 10 pips)
    THRESHOLD = 1.0 
    
    signal = "HOLD"
    confidence = 0.0
    
    # Calculate Confidence based on strength of move
    # Logic: Base 50%. Every $1 move adds ~20% confidence, capped at 99%.
    move_strength = abs(predicted_price - current_price)
    confidence = min(99.9, 50.0 + (move_strength * 20.0))
    
    if predicted_price > current_price + THRESHOLD:
        signal = "BUY"
    elif predicted_price < current_price - THRESHOLD:
        signal = "SELL"
    else:
        # If HOLD, confidence is how sure we are about holding (inverse of move strength?)
        pass
        
    return signal, predicted_price, confidence