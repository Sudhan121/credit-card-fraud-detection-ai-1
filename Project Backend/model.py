import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

class FraudModel:
    def __init__(self):
        # Load the pre-trained model and other necessary objects (e.g., scaler, selected features)
        self.model = joblib.load("model/xgboost_model.pkl")  # Load XGBoost model
        self.scaler = joblib.load("model/scaler.pkl")  # StandardScaler for feature scaling
        self.selected_features = joblib.load("model/selected_features.pkl")  # List of selected features

    def preprocess(self, df):
        """
        Preprocess the incoming dataframe.
        - Feature engineering like creating new columns or scaling.
        - Drop irrelevant or redundant features.
         """
        # Feature engineering: Extract the hour from the 'Time' column (if present)
        df['Hour'] = (df['Time'] // 3600) % 24
        df['Log_Amount'] = np.log(df['Amount'] + 1)  # Log-transform 'Amount' to stabilize variance

        # Drop unnecessary columns (e.g., 'Time', 'Amount' are replaced with derived features)
        df = df.drop(columns=['Time', 'Amount'], errors='ignore')

        # Ensure we have the same columns as the model was trained on
        df = df[self.selected_features]

        # Apply scaling
        df_scaled = self.scaler.transform(df)
        
        return df_scaled

    def predict(self, df):
        """
        Make predictions on the processed dataframe.
        - Predict using the pre-trained model.
        - Return predictions and probabilities.
        """
        processed_df = self.preprocess(df)  # Preprocess the dataframe first
        preds = self.model.predict(processed_df)  # Get class predictions (0 for legitimate, 1 for fraud)
        probs = self.model.predict_proba(processed_df)[:, 1]  # Get probability for fraud class (1)

        return preds, probs
