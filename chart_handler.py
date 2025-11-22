import plotly.graph_objects as go
import pandas as pd

def create_candlestick_chart(df, setup, symbol="XAUUSD"):
    """
    Creates a Plotly Candlestick chart with Entry, SL, and TP lines.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Date', 'Open', 'High', 'Low', 'Close'.
        setup (dict): Dictionary containing 'entry', 'sl', 'tp' for the active signal.
        symbol (str): The symbol name.
        
    Returns:
        go.Figure: The plotly figure.
    """
    fig = go.Figure()

    # 1. Candlestick Trace
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=symbol
    ))

    # 2. Add Lines if setup exists
    if setup and setup.get('entry') > 0:
        entry = setup['entry']
        sl = setup['sl']
        tp = setup['tp']
        
        # Entry Line (Blue)
        fig.add_hline(y=entry, line_dash="dash", line_color="blue", annotation_text="Entry", annotation_position="top right")
        
        # Stop Loss (Red)
        fig.add_hline(y=sl, line_dash="dash", line_color="red", annotation_text="SL", annotation_position="bottom right")
        
        # Take Profit (Green)
        fig.add_hline(y=tp, line_dash="dash", line_color="green", annotation_text="TP", annotation_position="top right")

    # 3. Layout Customization
    fig.update_layout(
        title=f"{symbol} Price Action",
        yaxis_title="Price",
        xaxis_title="Time",
        template="plotly_dark",
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False
    )

    return fig
