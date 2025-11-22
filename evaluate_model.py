import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import mean_absolute_error

# Import from model_training
from model_training import (
    load_data, add_technical_indicators, align_higher_timeframe, 
    preprocess_data, CSV_FILE_PATH_5M, CSV_FILE_PATH_15M, CSV_FILE_PATH_1H
)

print("--- EVALUATING ENHANCED MODEL ---")

# 1. Load and Process Data (Same as training)
print("\n--- Loading Data ---")
df_5m = load_data(CSV_FILE_PATH_5M)
if df_5m is None:
    exit()

df_15m = load_data(CSV_FILE_PATH_15M)
df_1h = load_data(CSV_FILE_PATH_1H)

# 2. Add indicators
print("\n--- Adding Technical Indicators ---")
df_5m = add_technical_indicators(df_5m, prefix='')

if df_15m is not None:
    df_15m = add_technical_indicators(df_15m, prefix='')
    df_5m = align_higher_timeframe(df_5m, df_15m, 'HTF15_')

if df_1h is not None:
    df_1h = add_technical_indicators(df_1h, prefix='')
    df_5m = align_higher_timeframe(df_5m, df_1h, 'HTF1H_')

# 3. Preprocess
print("\n--- Preprocessing ---")
X, y, _, _, _ = preprocess_data(df_5m)

# 4. Load Model
print("\n--- Loading Model ---")
model = load_model('best_xauusd_model.keras')
target_scaler = joblib.load('target_scaler.pkl')

# 5. Predict
print("\n--- Predicting ---")
predictions = model.predict(X)

# 6. Inverse Transform
predictions_actual = target_scaler.inverse_transform(predictions)
y_actual = target_scaler.inverse_transform(y)

# 7. Calculate Metrics on TEST SET (Last 20%)
split = int(len(X) * 0.8)
y_test = y_actual[split:]
pred_test = predictions_actual[split:]

# A. Mean Absolute Error (MAE) on Log Returns
mae = mean_absolute_error(y_test, pred_test)
print(f"\n{'='*60}")
print(f"EVALUATION RESULTS")
print(f"{'='*60}")
print(f"\nMean Absolute Error (MAE) on Log Returns: {mae:.6f}")
print(f"Interpretation: On average, the model's log return prediction is off by {mae:.6f}.")

# B. Directional Accuracy
correct_direction = 0
total = 0

for i in range(len(y_test)):
    actual_return = y_test[i][0]
    pred_return = pred_test[i][0]
    
    if (actual_return > 0 and pred_return > 0) or (actual_return < 0 and pred_return < 0):
        correct_direction += 1
    
    total += 1

accuracy = (correct_direction / total) * 100
print(f"\nDirectional Accuracy: {accuracy:.2f}%")
print(f"Interpretation: The model correctly predicts the UP/DOWN direction {accuracy:.2f}% of the time.")

# C. Convert Log Returns to Approximate Price Changes
current_price = 2915
avg_predicted_price_change = np.mean(np.abs(pred_test)) * current_price
avg_actual_price_change = np.mean(np.abs(y_test)) * current_price

print(f"\n{'='*60}")
print(f"PRICE CONTEXT")
print(f"{'='*60}")
print(f"Average Predicted Price Change: ${avg_predicted_price_change:.2f}")
print(f"Average Actual Price Change: ${avg_actual_price_change:.2f}")
print(f"Approximate Price Error: ${mae * current_price:.2f}")

# D. Performance Comparison
print(f"\n{'='*60}")
print(f"IMPROVEMENT vs BASELINE")
print(f"{'='*60}")
print(f"Baseline Model (9 features, no MTF):")
print(f"  - MAE: 0.000677 (~$1.97)")
print(f"  - Directional Accuracy: 51.29%")
print(f"\nEnhanced Model (40 features, with MTF):")
print(f"  - MAE: {mae:.6f} (~${mae * current_price:.2f})")
print(f"  - Directional Accuracy: {accuracy:.2f}%")

if mae < 0.000677:
    improvement = ((0.000677 - mae) / 0.000677) * 100
    print(f"\n✓ MAE Improved by {improvement:.1f}%")
else:
    degradation = ((mae - 0.000677) / 0.000677) * 100
    print(f"\n✗ MAE Degraded by {degradation:.1f}%")

if accuracy > 51.29:
    improvement = accuracy - 51.29
    print(f"✓ Directional Accuracy Improved by {improvement:.2f} percentage points")
else:
    degradation = 51.29 - accuracy
    print(f"✗ Directional Accuracy Degraded by {degradation:.2f} percentage points")

print(f"\n{'='*60}")
