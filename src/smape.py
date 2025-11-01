import pandas as pd
import numpy as np
import sys

def smape(actual, predicted):
    """Compute Symmetric Mean Absolute Percentage Error (SMAPE)."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    diff = np.abs(predicted - actual)

    # Avoid division by zero
    mask = denominator != 0
    smape_value = np.mean(diff[mask] / denominator[mask]) * 100
    return smape_value

def main(actual_file, predicted_file):
    # Read CSVs
    df_actual = pd.read_csv(actual_file)
    df_pred = pd.read_csv(predicted_file)

    # Ensure 'price' column exists
    if 'price' not in df_actual.columns or 'price' not in df_pred.columns:
        raise ValueError("Both CSV files must contain a 'price' column")
    
    actual_prices = df_actual['price']
    predicted_prices = df_pred['price']

    # Compute SMAPE
    result = smape(actual_prices, predicted_prices)
    print(f"✅ SMAPE: {result:.4f}%")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compute_smape.py actual.csv predicted.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
