import MetaTrader5 as mt5
import pandas as pd
import datetime

def initialize_mt5():
    """
    Initializes the connection to the MetaTrader 5 terminal.
    Returns True if successful, False otherwise.
    """
    # Explicit path found on user system
    mt5_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    
    # Credentials provided by user
    login = 99199654
    password = "OmK@Z7Sw"
    server = "MetaQuotes-Demo"

    # Initialize with path
    if not mt5.initialize(path=mt5_path):
        print("initialize() failed, error code =", mt5.last_error())
        return False
        
    # Login to account
    authorized = mt5.login(login, password=password, server=server)
    if not authorized:
        print("failed to connect at account #{}, error code: {}".format(login, mt5.last_error()))
        return False
    
    print(f"Connected to MetaTrader 5! Terminal: {mt5.terminal_info().name}")
    print(f"Account: {login} ({server})")
    return True

def shutdown_mt5():
    mt5.shutdown()

def get_mt5_data(symbol, timeframe_str, count=500):
    """
    Fetches historical data from MT5.
    
    Args:
        symbol (str): The symbol to fetch (e.g., "XAUUSD").
        timeframe_str (str): Timeframe string (e.g., "5m").
        count (int): Number of candles to fetch.
        
    Returns:
        pd.DataFrame: DataFrame with Open, High, Low, Close, Volume.
    """
    # Map string timeframe to MT5 constant
    tf_map = {
        "1m": mt5.TIMEFRAME_M1,
        "3m": mt5.TIMEFRAME_M3,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1
    }
    
    mt5_tf = tf_map.get(timeframe_str)
    if mt5_tf is None:
        print(f"Error: Invalid timeframe '{timeframe_str}'")
        return None

    # Fetch rates
    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
    
    if rates is None or len(rates) == 0:
        print(f"Error: Failed to fetch data for {symbol}. Error: {mt5.last_error()}")
        return None
        
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Rename columns to match our model's expectation
    df.rename(columns={
        'time': 'Date', 
        'open': 'Open', 
        'high': 'High', 
        'low': 'Low', 
        'close': 'Close', 
        'tick_volume': 'Volume'
    }, inplace=True)
    
    df.set_index('Date', inplace=True)
    
    # Keep only necessary columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    return df

def place_mt5_order(symbol, action, volume, sl=0.0, tp=0.0, deviation=20):
    """
    Places a market order in MT5.
    
    Args:
        symbol (str): Symbol to trade.
        action (str): "BUY" or "SELL".
        volume (float): Lot size.
        sl (float): Stop Loss price.
        tp (float): Take Profit price.
        deviation (int): Max slippage in points.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    # Check if symbol is available
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"{symbol} not found, can not call order_check()")
        return False
    
    if not symbol_info.visible:
        print(f"{symbol} is not visible, trying to switch on")
        if not mt5.symbol_select(symbol, True):
            print(f"symbol_select({symbol}) failed, exit")
            return False
            
    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if action == "BUY" else mt5.symbol_info_tick(symbol).bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": price,
        "sl": float(sl),
        "tp": float(tp),
        "deviation": deviation,
        "magic": 234000, # Magic number to identify bot trades
        "comment": "AI Bot Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Send order
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order Failed: {result.comment} (Code: {result.retcode})")
        return False
    
    print(f"Order Placed Successfully! Ticket: {result.order}")
    return True

def get_open_positions(symbol=None):
    """
    Checks for open positions.
    If symbol is provided, returns positions for that symbol only.
    Returns a list of positions (or empty list).
    """
    if symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()

    if positions is None:
        return []
    
    return positions
