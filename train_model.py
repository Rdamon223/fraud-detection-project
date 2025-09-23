import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# Load credit card set data
df = pd.read_csv('data/creditcard.csv')
print(df.head())  # Check first rows

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Handle imbalance (undersample majority class)
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data preprocessed!")

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# Train model
model = IsolationForest(contamination=0.0017, random_state=42)
model.fit(X_train)  # IsolationForest doesn't need y_train for fitting

# Evaluate (predict -1 for anomaly/fraud, 1 for normal)
y_pred = model.predict(X_test)
y_pred = [1 if x == -1 else 0 for x in y_pred]  # Map to 0/1 for metrics
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save model and scaler
import joblib
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model trained and saved!")