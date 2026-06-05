import pandas as pd
from datetime import datetime
import joblib
import numpy as np

# Models and Tools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
# Note: Evaluation metrics are no longer imported

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
df_processed['age'].fillna(df_processed['age'].median(), inplace=True)
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
# Note: We are not creating a separate test set here, as evaluation is not required.
# The entire dataset will be used to train the final model for best performance.

print("Applying SMOTE to balance the full dataset...")
smote = SMOTE(random_state=42)
X_res, y_res,*_ = smote.fit_resample(X, y)
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

# Train on the full, balanced dataset
ensemble_model.fit(X_res, y_res)
print("Ensemble model trained successfully.")

# --- 4. Save the Final Model ---
joblib.dump(ensemble_model, 'final_model.joblib')
print("Final ensemble model saved as 'final_model.joblib'.")


print("\n--- Training Script Finished Successfully ---")