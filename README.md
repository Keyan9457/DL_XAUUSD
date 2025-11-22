# AI-Powered XAUUSD Trading Bot

This project is a fully functional, AI-driven trading bot for Gold (XAUUSD). It uses Deep Learning (LSTM) combined with Smart Money Concepts (SMC) to predict prices and generate trade signals.

## 🚀 Features

### 1. AI Brain (LSTM + Regression)
-   **Model**: Long Short-Term Memory (LSTM) Neural Network.
-   **Prediction**: Predicts the **Exact Next Price** (Regression) rather than just Buy/Sell.
-   **Features**: Uses Technical Indicators (RSI, MACD, EMA, ATR, Bollinger Bands) AND **SMC Features** (Distance to Swing Highs/Lows).

### 2. Smart Trading Logic
-   **Trend Filter**: Strictly follows the **50 EMA**.
    -   **BUY**: Only if Price > 50 EMA.
    -   **SELL**: Only if Price < 50 EMA.
-   **SMC Stop Loss**: Dynamic SL placement based on Market Structure.
    -   **BUY SL**: Below the recent Swing Low.
    -   **SELL SL**: Above the recent Swing High.
-   **Take Profit**: Automatically set to **1:2 Risk-to-Reward**.
-   **Scalping Focus**: Designed for 5-15 minute trades (Scalps).

### 3. Real-Time Execution
-   **Live Data**: Fetches real-time data from Yahoo Finance (`yfinance`).
-   **Timezone Sync**: Automatically converts chart time to **IST (Indian Standard Time)**.
-   **Risk Management**: Calculates Lot Size based on Account Balance and Risk per Trade.

## 📂 File Structure

-   `main.py`: **The Engine**. Runs the live trading loop, fetches data, and executes trades.
-   `model_training.py`: **The Teacher**. Used to train or retrain the AI model.
-   `live_trader.py`: **The Manager**. Handles signal logic, risk calculations, and trade simulation.
-   `best_xauusd_model.keras`: The trained AI model file.
-   `scaler.pkl` / `target_scaler.pkl`: Saved scalers for data normalization.

## ⚡ How to Run

1.  **Start the Bot**:
    ```powershell
    python main.py
    ```
2.  **Monitor Output**:
    The bot will print the status every 5 minutes:
    ```text
    Time: 2025-11-21 06:45:00 (IST)
    Current Price: 4066.30
    Predicted Price: 4070.27
    Trend: BULLISH (Above EMA 50)
    Signal: BUY
    Est. Duration: 5-15 Mins (Scalp)
    Stop Loss: 4059.83
    Take Profit: 4083.14
    ```

## ⚠️ Important Notes
-   **Simulation Mode**: The bot currently *simulates* trades (prints them). To trade real money, integrate a broker API in `live_trader.py`.
-   **Data Delay**: `yfinance` data for Futures is delayed by 10-15 mins. For real trading, use a paid data feed.
