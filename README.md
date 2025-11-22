# XAUUSD AI Trading Bot

AI-powered trading bot for Gold (XAUUSD) using LSTM neural networks with Smart Money Concepts (SMC) and multi-timeframe analysis.

## 🚀 Features

- **Deep Learning**: LSTM neural network predicting price movements
- **Multi-Timeframe Analysis**: 5m, 15m, 1h, 4h, Daily timeframes
- **Smart Money Concepts**: Dynamic SL/TP based on market structure
- **55-Year Historical Data**: Training on comprehensive dataset (2004-2025)
- **Risk Management**: Automatic position sizing, 1:2 R:R ratio
- **Real-Time Execution**: MetaTrader 5 integration
- **WhatsApp Notifications**: Instant trade alerts

## 📊 Model Architecture

- **Input**: 120 candles × 60 features (multi-timeframe indicators)
- **Architecture**: 3-layer LSTM (256-256-128 units) with dropout & batch normalization
- **Output**: Log return prediction (regression)
- **Target**: 55%+ directional accuracy

## 🛠️ Installation

### Local Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/DL_XAUUSD.git
cd DL_XAUUSD

# Install dependencies
pip install -r requirements.txt

# Configure MT5 credentials
cp .env.example .env
# Edit .env with your MT5 login details
```

### Google Colab Training

For training with GPU acceleration:

1. Open `train_model_colab.ipynb` in Google Colab
2. Upload your historical data to Colab or Google Drive
3. Run all cells to train the model
4. Download trained model files

## 📁 Project Structure

```
DL_XAUUSD/
├── main.py                          # Live trading bot
├── model_training.py                # Model training script
├── evaluate_model.py                # Model evaluation
├── process_historical_data.py       # Data preprocessing
├── live_trader.py                   # Trading logic & risk management
├── mt5_handler.py                   # MetaTrader 5 integration
├── notification_handler.py          # WhatsApp alerts
├── dashboard.py                     # Streamlit dashboard
├── train_model_colab.ipynb         # Colab training notebook
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🎯 Usage

### 1. Process Historical Data

```bash
python process_historical_data.py
```

### 2. Train Model

**Local (if you have 16GB+ RAM):**
```bash
python model_training.py
```

**Google Colab (recommended):**
- Open `train_model_colab.ipynb`
- Follow notebook instructions

### 3. Evaluate Model

```bash
python evaluate_model.py
```

### 4. Run Live Trading

```bash
python main.py
```

### 5. Launch Dashboard

```bash
streamlit run dashboard.py
```

## 📈 Performance Metrics

**Target Metrics:**
- MAE on Log Returns: < 0.0005
- Directional Accuracy: > 55%
- Risk-Reward Ratio: 1:2

## ⚙️ Configuration

Edit `main.py` to configure:
- `SYMBOL`: Trading symbol (default: XAUUSD)
- `INTERVAL`: Timeframe (default: 5m)
- `ACCOUNT_BALANCE`: Your account size
- Risk percentage per trade

## 🔒 Security

- Never commit `.env` file (contains MT5 credentials)
- Keep API keys secure
- Use paper trading first to test

## 📝 System Architecture

The bot uses a "Waterfall" logic with 3 gates:

1. **Gate 1 (HTF Trend)**: 30m & 15m EMA confirmation
2. **Gate 2 (AI Prediction)**: LSTM price prediction on 5m
3. **Gate 3 (LTF Entry)**: 3m & 1m momentum confirmation

Only trades that pass all 3 gates are executed.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## ⚠️ Disclaimer

This bot is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly in a demo account before live trading.

## 📄 License

MIT License - see LICENSE file for details

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using TensorFlow, MetaTrader 5, and Smart Money Concepts**
