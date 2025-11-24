import joblib

scaler = joblib.load('scaler.pkl')

print(f"Total features expected: {scaler.n_features_in_}")
print("\nFeature names (if available):")
if hasattr(scaler, 'feature_names_in_'):
    for i, name in enumerate(scaler.feature_names_in_):
        print(f"{i+1:2d}. {name}")
else:
    print("No feature names stored in scaler")
