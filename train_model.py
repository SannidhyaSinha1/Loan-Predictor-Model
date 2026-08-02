import pandas as pd
from datetime import datetime
import joblib
import numpy as np
import json

# Models and Tools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

print("--- Starting Model Training ---")

# --- 1. Load, Clean, and Engineer Features ---
df = pd.read_csv('data.csv')
print("Cleaning data...")
df.drop_duplicates(inplace=True)
df.drop_duplicates(subset=['userId', 'loanId'], keep='first', inplace=True)
df_processed = df.drop(columns=['userId', 'loanId', 'fullName', 'gender', 'type', 'totalCreditCardCurrentBalance', 'totalCreditCardLimit'])
def calculate_age(born):
    try: return datetime.today().year - datetime.strptime(str(born), '%Y-%m-%d').year
    except: return None
df_processed['age'] = df_processed['dateOfBirth'].apply(calculate_age)
df_processed = df_processed.drop(columns=['dateOfBirth'])
df_processed['age'] = df_processed['age'].fillna(df_processed['age'].median())
median_score = df_processed[df_processed['creditScore'] > 0]['creditScore'].median()
df_processed['creditScore'] = df_processed['creditScore'].replace(0, median_score)
df_processed['isPayingTimelyEmis'] = df_processed['isPayingTimelyEmis'].map({'Yes': 1, 'No': 0})

print("Creating engineered features...")
df_processed['creditUtilizationPercentage'] = df_processed['creditUtilizationPercentage'] / 100.0
df_processed['distress_score'] = df_processed['creditUtilizationPercentage'] * (df_processed['loanWithLatePaymentCount'] + 1)
df_processed['secured_loan_ratio'] = df_processed['securedLoanCount'] / (df_processed['securedLoanCount'] + df_processed['unsecuredLoanCount'] + 1)
df_processed['debt_to_age_ratio'] = df_processed['unsecuredLoanCount'] / (df_processed['age'] + 1)
df_processed.dropna(inplace=True)

# --- 2. Data Splitting & Balancing ---
X = df_processed.drop('isPayingTimelyEmis', axis=1)
y = df_processed['isPayingTimelyEmis']

# IMPORTANT: split FIRST, then SMOTE only the training fold.
# Resampling before the split leaks synthetic neighbors of test-set points
# into the training data, which inflates test accuracy. The test set here
# stays untouched and reflects the real class imbalance.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print("Applying SMOTE to balance the TRAINING set only...")
smote = SMOTE(random_state=42)
X_res, y_res, *_ = smote.fit_resample(X_train, y_train)
print("SMOTE balancing complete.")

# --- 3. Define and Train the ENHANCED Ensemble Model ---
print("\nDefining and training the final ensemble model...")

clf1 = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='logloss')
clf2 = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=42)
clf3 = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(random_state=42))])

specialist_features = ['creditScore', 'creditUtilizationPercentage']
specialist_transformer = ColumnTransformer([('scaler', StandardScaler(), specialist_features)], remainder='drop')
clf4 = Pipeline([('selector', specialist_transformer), ('logreg', LogisticRegression(random_state=42))])

model_weights = [0.40, 0.40, 0.10, 0.10]

ensemble_model = VotingClassifier(
    estimators=[('xgb', clf1), ('rf', clf2), ('lr', clf3), ('cs_util_expert', clf4)],
    voting='soft', weights=model_weights
)

# Train on the SMOTE-balanced TRAINING set only
ensemble_model.fit(X_res, y_res)
print("Ensemble model trained successfully.")

# --- 4. Evaluate on the untouched, real-distribution test set ---
print("\n--- Evaluating on held-out test set (real class distribution, no leakage) ---")
y_pred = ensemble_model.predict(X_test)
y_proba = ensemble_model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1_score": round(f1_score(y_test, y_pred), 4),
}
cm = confusion_matrix(y_test, y_pred)

print("Test set metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")
print("Confusion matrix (rows=actual, cols=predicted, order=[No, Yes]):")
print(cm)

with open('metrics.json', 'w') as f:
    json.dump({**metrics, "confusion_matrix": cm.tolist()}, f, indent=2)
print("Metrics saved to 'metrics.json'.")

# --- 5. Retrain on the FULL balanced dataset for the model actually shipped ---
# The model evaluated above is trained only on the 80% training split so the
# test metrics are honest. For the model we deploy, refit on all data
# (train + test) after SMOTE, now that we already know its real performance.
print("\nRefitting final production model on the FULL SMOTE-balanced dataset...")
smote_full = SMOTE(random_state=42)
X_res_full, y_res_full, *_ = smote_full.fit_resample(X, y)
ensemble_model.fit(X_res_full, y_res_full)

# --- 6. Save the Final Model ---
joblib.dump(ensemble_model, 'final_model.joblib')
print("Final ensemble model saved as 'final_model.joblib'.")

print("\n--- Training Script Finished Successfully ---")