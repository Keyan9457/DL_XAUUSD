"""
Smart Money Concepts (SMC) Indicators
Implements FVG, Order Blocks, Breaker Blocks, BOS, and CHoCH
"""

import pandas as pd
import numpy as np

def detect_fvg(df):
    """
    Detect Fair Value Gaps (FVG)
    A FVG occurs when there's a gap between candle 1's high/low and candle 3's low/high
    """
    df['Bullish_FVG'] = False
    df['Bearish_FVG'] = False
    df['FVG_Top'] = np.nan
    df['FVG_Bottom'] = np.nan
    
    for i in range(2, len(df)):
        # Bullish FVG: Gap between candle[i-2].high and candle[i].low
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            df.loc[df.index[i], 'Bullish_FVG'] = True
            df.loc[df.index[i], 'FVG_Bottom'] = df['High'].iloc[i-2]
            df.loc[df.index[i], 'FVG_Top'] = df['Low'].iloc[i]
        
        # Bearish FVG: Gap between candle[i-2].low and candle[i].high
        if df['High'].iloc[i] < df['Low'].iloc[i-2]:
            df.loc[df.index[i], 'Bearish_FVG'] = True
            df.loc[df.index[i], 'FVG_Top'] = df['Low'].iloc[i-2]
            df.loc[df.index[i], 'FVG_Bottom'] = df['High'].iloc[i]
    
    return df

def detect_order_blocks(df, lookback=20):
    """
    Detect Order Blocks
    An Order Block is the last opposite-colored candle before a strong move
    """
    df['Bullish_OB'] = False
    df['Bearish_OB'] = False
    df['OB_Top'] = np.nan
    df['OB_Bottom'] = np.nan
    
    for i in range(lookback, len(df)):
        # Look for strong bullish move (3+ consecutive bullish candles)
        if all(df['Close'].iloc[i-j] > df['Open'].iloc[i-j] for j in range(3)):
            # Find last bearish candle before the move
            for j in range(4, lookback):
                if df['Close'].iloc[i-j] < df['Open'].iloc[i-j]:
                    df.loc[df.index[i-j], 'Bullish_OB'] = True
                    df.loc[df.index[i-j], 'OB_Bottom'] = df['Low'].iloc[i-j]
                    df.loc[df.index[i-j], 'OB_Top'] = df['High'].iloc[i-j]
                    break
        
        # Look for strong bearish move (3+ consecutive bearish candles)
        if all(df['Close'].iloc[i-j] < df['Open'].iloc[i-j] for j in range(3)):
            # Find last bullish candle before the move
            for j in range(4, lookback):
                if df['Close'].iloc[i-j] > df['Open'].iloc[i-j]:
                    df.loc[df.index[i-j], 'Bearish_OB'] = True
                    df.loc[df.index[i-j], 'OB_Bottom'] = df['Low'].iloc[i-j]
                    df.loc[df.index[i-j], 'OB_Top'] = df['High'].iloc[i-j]
                    break
    
    return df

def detect_breaker_blocks(df):
    """
    Detect Breaker Blocks
    A Breaker Block is a failed Order Block that changes polarity
    """
    df['Bullish_BB'] = False
    df['Bearish_BB'] = False
    
    # Find Order Blocks first
    bullish_obs = df[df['Bullish_OB'] == True].index
    bearish_obs = df[df['Bearish_OB'] == True].index
    
    for ob_idx in bullish_obs:
        ob_top = df.loc[ob_idx, 'OB_Top']
        ob_bottom = df.loc[ob_idx, 'OB_Bottom']
        
        # Check if price broke below the bullish OB (failed)
        future_data = df.loc[ob_idx:].iloc[1:]
        if any(future_data['Close'] < ob_bottom):
            # It's now a Bearish Breaker Block
            df.loc[ob_idx, 'Bearish_BB'] = True
            df.loc[ob_idx, 'Bullish_OB'] = False
    
    for ob_idx in bearish_obs:
        ob_top = df.loc[ob_idx, 'OB_Top']
        ob_bottom = df.loc[ob_idx, 'OB_Bottom']
        
        # Check if price broke above the bearish OB (failed)
        future_data = df.loc[ob_idx:].iloc[1:]
        if any(future_data['Close'] > ob_top):
            # It's now a Bullish Breaker Block
            df.loc[ob_idx, 'Bullish_BB'] = True
            df.loc[ob_idx, 'Bearish_OB'] = False
    
    return df

def detect_bos_choch(df, swing_window=10):
    """
    Detect Break of Structure (BOS) and Change of Character (CHoCH)
    BOS: Breaking previous high/low in trend direction
    CHoCH: Breaking previous high/low against trend (trend reversal)
    """
    df['BOS'] = 0  # 1 = Bullish BOS, -1 = Bearish BOS
    df['CHoCH'] = 0  # 1 = Bullish CHoCH, -1 = Bearish CHoCH
    
    # Identify swing highs and lows (without center=True to preserve data)
    df['Swing_High'] = df['High'].rolling(window=swing_window).max() == df['High']
    df['Swing_Low'] = df['Low'].rolling(window=swing_window).min() == df['Low']
    
    swing_highs = df[df['Swing_High'] == True]['High'].to_dict()
    swing_lows = df[df['Swing_Low'] == True]['Low'].to_dict()
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return df
    
    # Determine trend based on higher highs/higher lows or lower highs/lower lows
    trend = 'neutral'
    
    for i in range(swing_window, len(df)):
        current_price = df['Close'].iloc[i]
        
        # Get recent swing points
        recent_highs = [v for k, v in swing_highs.items() if k < df.index[i]][-2:]
        recent_lows = [v for k, v in swing_lows.items() if k < df.index[i]][-2:]
        
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            # Uptrend: Higher Highs and Higher Lows
            if recent_highs[-1] > recent_highs[-2] and recent_lows[-1] > recent_lows[-2]:
                trend = 'uptrend'
            # Downtrend: Lower Highs and Lower Lows
            elif recent_highs[-1] < recent_highs[-2] and recent_lows[-1] < recent_lows[-2]:
                trend = 'downtrend'
            
            # Detect BOS (continuation)
            if trend == 'uptrend' and current_price > recent_highs[-1]:
                df.loc[df.index[i], 'BOS'] = 1
            elif trend == 'downtrend' and current_price < recent_lows[-1]:
                df.loc[df.index[i], 'BOS'] = -1
            
            # Detect CHoCH (reversal)
            if trend == 'uptrend' and current_price < recent_lows[-1]:
                df.loc[df.index[i], 'CHoCH'] = -1
            elif trend == 'downtrend' and current_price > recent_highs[-1]:
                df.loc[df.index[i], 'CHoCH'] = 1
    
    return df

def get_smc_signal(df, current_price):
    """
    Generate trading signal based on SMC indicators
    Returns: signal (BUY/SELL/HOLD), confidence (0-1), reason
    """
    if len(df) < 20:
        return "HOLD", 0.0, "Insufficient data"
    
    latest = df.iloc[-1]
    signal = "HOLD"
    confidence = 0.0
    reasons = []
    
    # Check for Bullish setups
    bullish_score = 0
    
    # 1. Price near Bullish FVG
    if not pd.isna(latest['FVG_Bottom']) and latest['Bullish_FVG']:
        if abs(current_price - latest['FVG_Bottom']) / current_price < 0.001:  # Within 0.1%
            bullish_score += 2
            reasons.append("Price at Bullish FVG")
    
    # 2. Price near Bullish Order Block
    if latest['Bullish_OB'] and not pd.isna(latest['OB_Bottom']):
        if latest['OB_Bottom'] <= current_price <= latest['OB_Top']:
            bullish_score += 3
            reasons.append("Price in Bullish OB")
    
    # 3. Bullish Breaker Block
    if latest['Bullish_BB']:
        bullish_score += 2
        reasons.append("Bullish Breaker Block")
    
    # 4. Bullish BOS (trend continuation)
    if latest['BOS'] == 1:
        bullish_score += 2
        reasons.append("Bullish BOS")
    
    # 5. Bullish CHoCH (trend reversal)
    if latest['CHoCH'] == 1:
        bullish_score += 3
        reasons.append("Bullish CHoCH")
    
    # Check for Bearish setups
    bearish_score = 0
    
    # 1. Price near Bearish FVG
    if not pd.isna(latest['FVG_Top']) and latest['Bearish_FVG']:
        if abs(current_price - latest['FVG_Top']) / current_price < 0.001:
            bearish_score += 2
            reasons.append("Price at Bearish FVG")
    
    # 2. Price near Bearish Order Block
    if latest['Bearish_OB'] and not pd.isna(latest['OB_Top']):
        if latest['OB_Bottom'] <= current_price <= latest['OB_Top']:
            bearish_score += 3
            reasons.append("Price in Bearish OB")
    
    # 3. Bearish Breaker Block
    if latest['Bearish_BB']:
        bearish_score += 2
        reasons.append("Bearish Breaker Block")
    
    # 4. Bearish BOS
    if latest['BOS'] == -1:
        bearish_score += 2
        reasons.append("Bearish BOS")
    
    # 5. Bearish CHoCH
    if latest['CHoCH'] == -1:
        bearish_score += 3
        reasons.append("Bearish CHoCH")
    
    # Determine signal
    if bullish_score > bearish_score and bullish_score >= 3:
        signal = "BUY"
        confidence = min(bullish_score / 10.0, 1.0)
    elif bearish_score > bullish_score and bearish_score >= 3:
        signal = "SELL"
        confidence = min(bearish_score / 10.0, 1.0)
    
    reason = ", ".join(reasons) if reasons else "No clear SMC setup"
    
    return signal, confidence, reason

def add_all_smc_indicators(df):
    """
    Add all SMC indicators to dataframe
    """
    df = detect_fvg(df)
    df = detect_order_blocks(df)
    df = detect_breaker_blocks(df)
    df = detect_bos_choch(df)
    
    # --- Fix for Data Loss ---
    # Forward fill price levels (carry forward last known level)
    price_cols = ['FVG_Top', 'FVG_Bottom', 'OB_Top', 'OB_Bottom']
    df[price_cols] = df[price_cols].ffill()
    
    # Fill remaining NaNs with 0 (for the beginning of data)
    # This prevents dropna() from removing all rows
    df.fillna(0, inplace=True)
    
    return df
