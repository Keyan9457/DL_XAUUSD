import streamlit as st
import json
import time
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(
    page_title="XAUUSD AI Trader Build by Passion and Made by Karthik - Supreme Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #41424b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .signal-box {
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .signal-buy { background-color: #00c853; color: white; }
    .signal-sell { background-color: #d50000; color: white; }
    .signal-neutral { background-color: #757575; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCTIONS ---
def load_state():
    try:
        with open('bot_state.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None

def get_signal_color(signal):
    if signal == "BUY" or signal == "BULLISH":
        return "signal-buy"
    elif signal == "SELL" or signal == "BEARISH":
        return "signal-sell"
    else:
        return "signal-neutral"

# --- MAIN LAYOUT ---
st.title("🤖 XAUUSD AI Live Trader")

# Auto-refresh logic
if 'last_run' not in st.session_state:
    st.session_state.last_run = time.time()

placeholder = st.empty()

while True:
    state = load_state()
    
    with placeholder.container():
        if state is None:
            st.warning("⚠️ Waiting for Bot State... (Ensure main.py is running)")
        else:
            # Header Metrics
            last_update = state.get('last_update', 'N/A')
            symbol = state.get('symbol', 'XAUUSD')
            price = state.get('current_price', 0.0)
            pred_price = state.get('predicted_price', 0.0)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Symbol", symbol)
            with col2:
                st.metric("Current Price", f"${price:,.2f}")
            with col3:
                delta = pred_price - price
                st.metric("AI Prediction", f"${pred_price:,.2f}", delta=f"{delta:.2f}")
            with col4:
                st.metric("Last Update", last_update)

            st.markdown("---")

            # Signals Section
            st.subheader("📡 Signal Analysis")
            
            signals = state.get('signals', {})
            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            
            with s_col1:
                st.markdown("**HTF Bias (30m/15m)**")
                val = signals.get('htf_bias', 'NEUTRAL')
                st.markdown(f'<div class="signal-box {get_signal_color(val)}">{val}</div>', unsafe_allow_html=True)
                
            with s_col2:
                st.markdown("**AI Model (5m)**")
                val = signals.get('ai_signal', 'HOLD')
                conf = signals.get('confidence', 0.0)
                st.markdown(f'<div class="signal-box {get_signal_color(val)}">{val}<br><span style="font-size:0.8em; opacity:0.8">{conf:.1f}%</span></div>', unsafe_allow_html=True)
                
            with s_col3:
                st.markdown("**LTF Conf (3m/1m)**")
                val = signals.get('ltf_conf', 'NEUTRAL')
                st.markdown(f'<div class="signal-box {get_signal_color(val)}">{val}</div>', unsafe_allow_html=True)
                
            with s_col4:
                st.markdown("**FINAL DECISION**")
                val = signals.get('final_signal', 'HOLD')
                st.markdown(f'<div class="signal-box {get_signal_color(val)}">{val}</div>', unsafe_allow_html=True)

            st.markdown("---")

            # Trade Setup Section
            st.subheader("🎯 Potential Trade Setup")
            
            setup = state.get('trade_setup', {})
            buy_setup = setup.get('buy', {})
            sell_setup = setup.get('sell', {})
            
            t_col1, t_col2 = st.columns(2)
            
            with t_col1:
                st.info("🔵 **Potential BUY Setup**")
                st.write(f"**Entry:** ${buy_setup.get('entry', 0):.2f}")
                st.write(f"**SL:** ${buy_setup.get('sl', 0):.2f}")
                st.write(f"**TP:** ${buy_setup.get('tp', 0):.2f}")
                
            with t_col2:
                st.error("🔴 **Potential SELL Setup**")
                st.write(f"**Entry:** ${sell_setup.get('entry', 0):.2f}")
                st.write(f"**SL:** ${sell_setup.get('sl', 0):.2f}")
                st.write(f"**TP:** ${sell_setup.get('tp', 0):.2f}")

            st.markdown("---")

            # Chart Section
            st.subheader("📊 Live Chart")
            
            chart_data = state.get('chart_data', [])
            if chart_data:
                from chart_handler import create_candlestick_chart
                
                # Convert back to DataFrame
                df_chart = pd.DataFrame(chart_data)
                
                # Determine which setup to show
                active_setup = {}
                final_signal = signals.get('final_signal', 'HOLD')
                
                if final_signal == "BUY":
                    active_setup = buy_setup
                elif final_signal == "SELL":
                    active_setup = sell_setup
                else:
                    # Show potential buy setup by default if neutral, or nothing
                    # Let's show the one that is closer or just buy for visualization
                    active_setup = buy_setup 

                fig = create_candlestick_chart(df_chart, active_setup, symbol)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Waiting for chart data...")

    # Sleep for 5 seconds before refresh
    time.sleep(5)
