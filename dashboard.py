import streamlit as st
import json
import time
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(
    page_title="XAUUSD AI Trader",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif;
        box-sizing: border-box;
    }
    
    /* Global App Styling */
    .stApp {
        background: radial-gradient(circle at top left, #1a1c2e 0%, #0f1016 100%);
        color: #ffffff;
    }
    
    /* Remove default Streamlit padding to fit single page */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* --- GLASSMORPHISM CARD SYSTEM --- */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    /* --- TYPOGRAPHY & COLORS --- */
    .label-text {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .value-text {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
    }
    
    .value-large {
        font-size: 2.2rem;
        background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .accent-cyan { color: #22d3ee; text-shadow: 0 0 15px rgba(34, 211, 238, 0.4); }
    .accent-green { color: #4ade80; text-shadow: 0 0 15px rgba(74, 222, 128, 0.4); }
    .accent-red { color: #f87171; text-shadow: 0 0 15px rgba(248, 113, 113, 0.4); }
    .accent-purple { color: #c084fc; text-shadow: 0 0 15px rgba(192, 132, 252, 0.4); }
    
    /* --- HEADER --- */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        padding: 0 0.5rem;
    }
    
    .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #22d3ee, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    
    .live-badge {
        background: rgba(74, 222, 128, 0.15);
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.3);
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .live-dot {
        width: 6px;
        height: 6px;
        background-color: #4ade80;
        border-radius: 50%;
        box-shadow: 0 0 8px #4ade80;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* --- SIGNAL BOXES --- */
    .signal-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .signal-item {
        text-align: center;
        padding: 1rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* --- TRADE SETUP --- */
    .setup-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
    }
    
    .trade-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .trade-row:last-child { border-bottom: none; }
    
    .trade-label { font-size: 0.85rem; color: #94a3b8; }
    .trade-val { font-family: 'Outfit', monospace; font-weight: 600; font-size: 0.95rem; }
    
    </style>
""", unsafe_allow_html=True)

# --- FUNCTIONS ---
def load_state():
    try:
        with open('bot_state.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def get_color_class(signal):
    if signal in ["BUY", "BULLISH"]: return "accent-green"
    if signal in ["SELL", "BEARISH"]: return "accent-red"
    return "label-text" # Neutral

# --- MAIN LAYOUT ---

# Auto-refresh
if 'last_run' not in st.session_state:
    st.session_state.last_run = time.time()

state = load_state()

if state is None:
    st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 3rem;">
            <h2 class="accent-red">⚠️ System Offline</h2>
            <p class="label-text">Waiting for main.py to start...</p>
        </div>
    """, unsafe_allow_html=True)
else:
    # Data Extraction
    symbol = state.get('symbol', 'XAUUSD')
    price = state.get('current_price', 0.0)
    pred_price = state.get('predicted_price', 0.0)
    last_update = state.get('last_update', 'N/A').split()[1] if ' ' in state.get('last_update', '') else state.get('last_update', 'N/A')
    signals = state.get('signals', {})
    setup = state.get('trade_setup', {})
    
    # 1. HEADER
    st.markdown(f"""
        <div class="header-container">
            <div class="app-title">💎 XAUUSD AI TRADER</div>
            <div class="live-badge">
                <div class="live-dot"></div>
                LIVE MARKET
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 2. HERO METRICS (Price & Prediction)
    col1, col2, col3, col4 = st.columns([1.5, 1.5, 1, 1])
    
    with col1:
        st.markdown(f"""
            <div class="glass-card">
                <div class="label-text">Current Price</div>
                <div class="value-text value-large">${price:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        delta = pred_price - price
        delta_sign = "+" if delta > 0 else ""
        delta_color = "accent-green" if delta > 0 else "accent-red"
        st.markdown(f"""
            <div class="glass-card">
                <div class="label-text">AI Prediction</div>
                <div class="value-text value-large" style="font-size: 1.8rem;">${pred_price:,.2f}</div>
                <div class="{delta_color}" style="font-size: 0.9rem; margin-top: 0.2rem;">
                    {delta_sign}{delta:.2f} (Delta)
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
            <div class="glass-card" style="align-items: center;">
                <div class="label-text">Symbol</div>
                <div class="value-text accent-purple">{symbol}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
            <div class="glass-card" style="align-items: center;">
                <div class="label-text">Last Update</div>
                <div class="value-text" style="font-size: 1.2rem;">{last_update}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    
    # 3. SIGNAL GRID
    st.markdown('<div class="label-text" style="margin-left: 0.5rem;">📡 Signal Analysis</div>', unsafe_allow_html=True)
    
    htf = signals.get('htf_bias', 'NEUTRAL')
    ai = signals.get('ai_signal', 'HOLD')
    conf = signals.get('confidence', 0.0)
    ltf = signals.get('ltf_conf', 'NEUTRAL')
    final = signals.get('final_signal', 'HOLD')
    
    st.markdown(f"""
        <div class="signal-grid">
            <div class="glass-card signal-item">
                <div class="label-text">HTF Bias</div>
                <div class="value-text {get_color_class(htf)}">{htf}</div>
            </div>
            <div class="glass-card signal-item">
                <div class="label-text">AI Model</div>
                <div class="value-text {get_color_class(ai)}">{ai}</div>
                <div class="label-text" style="margin-top:0.3rem; font-size: 0.65rem;">{conf:.1f}% Conf</div>
            </div>
            <div class="glass-card signal-item">
                <div class="label-text">LTF Conf</div>
                <div class="value-text {get_color_class(ltf)}">{ltf}</div>
            </div>
            <div class="glass-card signal-item" style="border-color: rgba(255,255,255,0.2);">
                <div class="label-text">Final Decision</div>
                <div class="value-text {get_color_class(final)}">{final}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 4. TRADE SETUPS
    st.markdown('<div class="label-text" style="margin-left: 0.5rem;">🎯 Active Setups</div>', unsafe_allow_html=True)
    
    buy = setup.get('buy', {})
    sell = setup.get('sell', {})
    
    col_buy, col_sell = st.columns(2)
    
    with col_buy:
        st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #4ade80;">
                <div style="display:flex; justify-content:space-between; margin-bottom:1rem;">
                    <span class="value-text accent-green" style="font-size: 1.2rem;">BUY SETUP</span>
                    <span class="label-text">LONG</span>
                </div>
                <div class="trade-row">
                    <span class="trade-label">ENTRY</span>
                    <span class="trade-val accent-cyan">${buy.get('entry', 0):.2f}</span>
                </div>
                <div class="trade-row">
                    <span class="trade-label">STOP LOSS</span>
                    <span class="trade-val accent-red">${buy.get('sl', 0):.2f}</span>
                </div>
                <div class="trade-row">
                    <span class="trade-label">TAKE PROFIT</span>
                    <span class="trade-val accent-green">${buy.get('tp', 0):.2f}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_sell:
        st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #f87171;">
                <div style="display:flex; justify-content:space-between; margin-bottom:1rem;">
                    <span class="value-text accent-red" style="font-size: 1.2rem;">SELL SETUP</span>
                    <span class="label-text">SHORT</span>
                </div>
                <div class="trade-row">
                    <span class="trade-label">ENTRY</span>
                    <span class="trade-val accent-cyan">${sell.get('entry', 0):.2f}</span>
                </div>
                <div class="trade-row">
                    <span class="trade-label">STOP LOSS</span>
                    <span class="trade-val accent-red">${sell.get('sl', 0):.2f}</span>
                </div>
                <div class="trade-row">
                    <span class="trade-label">TAKE PROFIT</span>
                    <span class="trade-val accent-green">${sell.get('tp', 0):.2f}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Auto-refresh
time.sleep(1)
st.rerun()
