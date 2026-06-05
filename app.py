# app.py

import streamlit as st
import requests
import json

# --- Page Config ---
st.set_page_config(page_title="Loan Predictor UI", page_icon="🧑‍💻", layout="wide")

# --- API Configuration ---
st.sidebar.header("API Configuration")
st.sidebar.info("Ensure your API service is running.")
api_url = st.sidebar.text_input("API Base URL", "http://127.0.0.1:8000")
API_KEY = "My_LoanApp_key_tmgN30rJ"

# --- Main UI ---
st.title('Loan Predictor Interface')
st.markdown("This UI sends data to a machine learning API to get predictions.")
st.markdown("---")

# --- User Input Form ---
with st.form("prediction_form"):
    st.header("Applicant Details")
    c1, c2 = st.columns(2)
    with c1:
        credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=700)
        credit_utilization_pct = st.number_input('Credit Utilization (%)', min_value=0.0, max_value=200.0, value=30.0, help="Enter as a percentage, e.g., 30 for 30%")
        late_payments = st.number_input('Loans with any Late Payments', min_value=0, max_value=100, value=0)
        closed_account_count = st.number_input('Number of Closed Accounts', min_value=0, max_value=100, value=5)
    with c2:
        secured_loans = st.number_input('Number of Secured Loans', min_value=0, max_value=100, value=1)
        unsecured_loans = st.number_input('Number of Unsecured Loans', min_value=0, max_value=100, value=3)
        age = st.number_input('Your Age', min_value=18, max_value=100, value=35)
        credit_card_loan_account = st.number_input('Number of Credit Card Loan Accounts', min_value=0, max_value=100, value=2)

    submitted = st.form_submit_button('✨ Get Prediction')

# --- Logic to Handle Form Submission ---
if submitted:
    if not api_url:
        st.error("🚫 Please enter a valid API URL in the sidebar.")
    else:
        api_data = {
            "credit_score": credit_score, "credit_utilization": credit_utilization_pct / 100.0,
            "late_payments": late_payments, "secured_loans": secured_loans,
            "unsecured_loans": unsecured_loans, "age": age,
            "credit_card_loan_account": credit_card_loan_account,
            "closed_account_count": closed_account_count
        }
        headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
        try:
            predict_url = f"{api_url.rstrip('/')}/predict"
            with st.spinner('Getting prediction...'):
                response = requests.post(predict_url, headers=headers, data=json.dumps(api_data))
            st.markdown("---")
            st.header("Prediction Result from API")
            if response.status_code == 200:
                results = response.json()
                final_proba = results.get("final_confidence_score", 0.0)
                if results.get("prediction_is_yes"):
                    st.success("## Yes, likely to pay on time.")
                else:
                    st.error("## No, high risk of missing payments.")
                st.metric("Confidence Score (Likelihood of Timely Repayment)", f"{final_proba:.1%}")
                st.progress(final_proba)
                with st.expander("Show Details"):
                    st.info(f"**Analysis Note:** {results.get('analysis', 'N/A')}")
                    st.subheader("Input Features Sent to API")
                    st.json(results.get('input_features', {}))
            else:
                st.error(f"Failed to get prediction. API returned status code {response.status_code}:")
                try:
                    st.json(response.json())
                except json.JSONDecodeError:
                    st.text(response.text)
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the API at `{api_url}`. Please check the URL and ensure the API server is running. Error: {e}")