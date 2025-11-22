import pandas as pd
from chart_handler import create_candlestick_chart
import plotly.io as pio

print("--- TESTING CHART HANDLER ---")

# 1. Create Dummy Data
data = {
    'Date': pd.date_range(start='2025-01-01', periods=50, freq='5min'),
    'Open': [2000 + i for i in range(50)],
    'High': [2005 + i for i in range(50)],
    'Low': [1995 + i for i in range(50)],
    'Close': [2002 + i for i in range(50)]
}
df = pd.DataFrame(data)

# 2. Create Dummy Setup
setup = {
    'entry': 2040,
    'sl': 2030,
    'tp': 2060
}

# 3. Generate Chart
try:
    fig = create_candlestick_chart(df, setup, "TEST_SYMBOL")
    print("Chart generated successfully.")
    
    # Optional: Save to HTML to inspect manually if needed
    # pio.write_html(fig, file='test_chart.html', auto_open=False)
    # print("Saved test_chart.html")
    
except Exception as e:
    print(f"Error generating chart: {e}")
