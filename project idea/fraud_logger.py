import pandas as pd

def log_fraudulent_transactions(transactions, predictions, output_file="fraud_report.csv"):
    """
    Logs fraudulent transactions to a CSV file.
    
    Parameters:
    - transactions (pd.DataFrame): Original transaction data.
    - predictions (np.array or list): Model predictions (0 = Legitimate, 1 = Fraud).
    - output_file (str): File path to save fraudulent transaction records.
    """
    # Combine predictions with transaction data
    transactions = transactions.copy()
    transactions['Prediction'] = predictions

    # Filter only fraudulent transactions
    fraud_df = transactions[transactions['Prediction'] == 1]

    # Save to CSV
    fraud_df.to_csv(output_file, index=False)
    print(f"Fraudulent transactions logged to {output_file}")
