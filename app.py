import streamlit as st
import requests
import os

st.set_page_config(page_title="Customer Prediction App", layout="wide")

API_URL = "https://customer-retention-fbg7.onrender.com"

st.title("📊 Customer Prediction App")
st.markdown("Enter customer data below to generate a real-time risk assessment.")

with st.form("user_inputs"):
    col1, col2, col3 =st.columns(3)

    with col1:
        st.subheader("Customer Details")
        tenure = st.slider("Tenure (Months)", min_value=1, max_value=12, value=6)
        monthly = st.number_input("Monthly Charges($)", min_value=0.0, value=70.0)
        total = st.number_input("Total Charges($)", min_value=0.0, value=420.0)
    
    with col2:
        st.subheader("Subscription Details")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        tech_support = st.checkbox("Has Tech Support?", ["Yes", "No"])
        online_security = st.checkbox("Has Online Security?", ["Yes", "No"])

    with col3:
        st.subheader("Demographics")
        gender = st.radio("Gender", ["Male", "Female"])
        senior = st.radio("Senior Citizen", ["Yes", "No"])
        partner = st.radio("Has Partner", ["Yes", "No"])

    submit = st.form_submit_button("Generate Prediction", use_container_width=True)

if submit:
    payload = {
        "Tenure_Months": tenure,
        "Monthly_Charges": monthly,
        "Total_Charges": total,
        "Contract_Month_to_month": 1 if contract == "Month-to-month" else 0,
        "Contract_One_year": 1 if contract == "One year" else 0,
        "Tech_Support_No": 0 if tech_support == "No" else 1,
        "Online_Security_No": 0 if online_security == "No" else 1,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Senior_Citizen_Yes": 1 if senior == "Yes" else 0,
        "Partner_Yes": 1 if partner == "Yes" else 0
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            
        if response.status_code == 200:
            data = response.json()["prediction"]

            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Churn Probability", data['probability'])
            c2.metric("Risk Assessment", data['risk'])
                                    
            if "High" in data ["risk"]:
                    st.error(f"**Action Required:**\n\n> {data['action']}")
            else:
                    st.success(f"**Recommendation:**\n\n> {data['action']}")

        else:
             st.error(f"API Error: {response.text}")    
                
    except Exception as e:
         st.error(f"Could not connect to API: {e}")

st.markdown("---")
st.caption("Backend Engine: FastAPI | ML Model: Random Forest | UI: Streamlit")