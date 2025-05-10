import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

# Load dataset
df = pd.read_csv('creditcard.csv')

# Feature Engineering
df['Hour'] = (df['Time'] // 3600) % 24
df['Log_Amount'] = np.log(df['Amount'] + 1)
df = df.drop(['Time', 'Amount'], axis=1)

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle imbalance with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# Train model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save model and preprocessing tools
joblib.dump(model, 'model/xgboost_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(X.columns.tolist(), 'model/selected_features.pkl')

print("✔️ Fraud detection model and assets saved successfully.")
