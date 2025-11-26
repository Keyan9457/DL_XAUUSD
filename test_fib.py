import pandas as pd
import numpy as np
from smc_indicators import get_smc_signal

def test_fib_logic():
    print("Testing Fibonacci Logic...")
    
    # Create a dummy dataframe
    # Scenario: Uptrend, then pullback to Golden Zone
    # Swing Low at 100, Swing High at 200
    # Golden Zone (Buy) is around 138.2 (61.8% retracement from 200) to 150 (50%)
    
    data = {
        'Open': [100] * 20,
        'High': [100] * 20,
        'Low': [100] * 20,
        'Close': [100] * 20,
        'Last_Swing_High': [200] * 20,
        'Last_Swing_Low': [100] * 20,
        'Bullish_FVG': [False] * 20,
        'Bearish_FVG': [False] * 20,
        'Bullish_OB': [False] * 20,
        'Bearish_OB': [False] * 20,
        'Bullish_BB': [False] * 20,
        'Bearish_BB': [False] * 20,
        'BOS': [0] * 20,
        'CHoCH': [0] * 20,
        'FVG_Bottom': [np.nan] * 20,
        'FVG_Top': [np.nan] * 20,
        'OB_Bottom': [np.nan] * 20,
        'OB_Top': [np.nan] * 20,
    }
    
    df = pd.DataFrame(data)
    
    # Set up a Bullish Signal context (e.g. Bullish OB) so we can test the Fib boost
    # Without a base signal, Fib logic might not trigger (it boosts existing signals)
    # Wait, looking at the code:
    # "if bullish_score > bearish_score: # Potential BUY"
    # So we need at least some bullish score.
    
    # Let's add a Bullish OB signal
    df.loc[19, 'Bullish_OB'] = True
    df.loc[19, 'OB_Bottom'] = 140
    df.loc[19, 'OB_Top'] = 145
    
    # Case 1: Price in Golden Zone (e.g. 140)
    # Range = 200 - 100 = 100
    # Retracement from High = (200 - 140) / 100 = 0.60 (60%) -> Golden Zone!
    current_price = 140
    
    signal, conf, reason = get_smc_signal(df, current_price)
    
    print(f"\nCase 1: Price {current_price} (Golden Zone)")
    print(f"Signal: {signal}")
    print(f"Confidence: {conf}")
    print(f"Reason: {reason}")
    
    if "Golden Zone Fib" in reason:
        print("✅ PASS: Golden Zone detected")
    else:
        print("❌ FAIL: Golden Zone NOT detected")

    # Case 2: Price in Premium Zone (e.g. 190)
    # Retracement = (200 - 190) / 100 = 0.10 (10%) -> Premium (Bad for Buy)
    # We need a base bullish signal to trigger the check. Let's say we have a BOS.
    df.loc[19, 'BOS'] = 1 
    
    current_price = 190
    signal, conf, reason = get_smc_signal(df, current_price)
    
    print(f"\nCase 2: Price {current_price} (Premium Zone)")
    print(f"Signal: {signal}")
    print(f"Confidence: {conf}")
    print(f"Reason: {reason}")
    
    if "Premium Zone" in reason:
        print("✅ PASS: Premium Zone warning detected")
    else:
        print("❌ FAIL: Premium Zone warning NOT detected")

if __name__ == "__main__":
    test_fib_logic()
