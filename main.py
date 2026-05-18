from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import BaseModel, Field, validator
import joblib
import os
import uvicorn

# 1. THE BLUEPRINT
class ChurnPredictor:
    def __init__(self, model, scaler, features):
        self.model = model
        self.scaler = scaler
        self.features = features

    def predict_and_recommend(self, input_data):
        if self.model is None:
            return [{"probability": 0.0, "recommendation": "Model weights missing"}]
        
        
        try:
            
            import pandas as pd
            df = pd.DataFrame(input_data)

            probs = self.model.predict_proba(df)[:, 1] 
            
            results = []
            for p in probs:
                if p > 0.8:
                    rec = "Urgent: High-priority retention call & 50% discount offer"
                elif p > 0.5:
                    rec = "Offer 20% discount & personalized feedback survey"
                else:
                    rec = "Standard follow-up: Send newsletter & feature updates"

                results.append({
                    "probability": float(p),
                    "recommendation": rec
                })

            return results
        except Exception as e:
            print(f"Internal Model Error: {e}")
            return [{"probability": 0.0, "recommendation": str(e)}]

# 2. THE NAMESPACE TRICK
import __main__
__main__.ChurnPredictor = ChurnPredictor

# 3. GLOBAL INITIALIZATION (Outside the __main__ block)
app = FastAPI(title="Customer Retention Engine")
MODEL_PATH = r'C:\Users\USER\customer_retention\retention_engine_v1.pkl'
engine = None
model_status = "Not loaded"

# THIS RUNS ON IMPORT (Uvicorn sees this)
if os.path.exists(MODEL_PATH):
    try:
        engine = joblib.load(MODEL_PATH)
        model_status = "Model loaded successfully."
        print(f"--- SUCCESS: {model_status} ---")
    except Exception as e:
        model_status = f"Load error: {e}"
        print(f"--- ERROR: {model_status} ---")
else:
    model_status = "File not found"
    print(f"--- ERROR: {model_status} ---")

class Customer(BaseModel):
    Tenure_Months: Optional[int] = Field(default=0)
    Monthly_Charges: Optional[float] = Field(default=0.0)
    Total_Charges: Optional[float] = Field(default=0.0)
    Contract_Month_to_month: Optional[int] = Field(default=0)
    Contract_One_year: Optional[int] = Field(default=0)
    Tech_Support_No: Optional[int] = Field(default=0)
    Online_Security_No: Optional[int] = Field(default=0)
    Online_Backup_No: Optional[int] = Field(default=0)
    Device_Protection_No: Optional[int] = Field(default=0)
    Streaming_TV_No: Optional[int] = Field(default=0)
    Streaming_Movies_No: Optional[int] = Field(default=0)

    Gender_Male: Optional[int] = Field(default=0)
    Senior_Citizen_Yes: Optional[int] = Field(default=0)
    Partner_Yes: Optional[int] = Field(default=0)
    Dependents_Yes: Optional[int] = Field(default=0)
    Paperless_Billing_Yes: Optional[int] = Field(default=0)
    Payment_Method_Electronic_check: Optional[int] = Field(default=0)

    @validator('*', pre=True)
    def check_empty_values(cls, v):
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return None # Pydantic will then use the 'default' value
        return v

@app.get("/")
def home():
    return {"status": "success", "model_status": model_status}

@app.post("/predict")
def predict_churn(data: Customer):
    
    if engine is None:
        raise HTTPException(status_code=500, detail=model_status)

    try:
        input_dict = data.dict()
        if input_dict['Total_Charges'] <= 0 and input_dict['Tenure_Months'] > 0:
            input_dict['Total_Charges'] = input_dict['Monthly_Charges'] * input_dict['Tenure_Months']
       
        features = {col: 0 for col in [
            'Tenure_Months', 'Monthly_Charges', 'Total_Charges', 'Gender_Female', 'Gender_Male', 
            'Senior_Citizen_No', 'Senior_Citizen_Yes', 'Partner_No', 'Partner_Yes', 'Dependents_No', 
            'Dependents_Yes', 'Phone_Service_No', 'Phone_Service_Yes', 'Multiple_Lines_No', 
            'Multiple_Lines_No_phone_service', 'Multiple_Lines_Yes', 'Internet_Service_DSL', 
            'Internet_Service_Fiber_optic', 'Internet_Service_No', 'Online_Security_No', 
            'Online_Security_No_internet_service', 'Online_Security_Yes', 'Online_Backup_No', 
            'Online_Backup_No_internet_service', 'Online_Backup_Yes', 'Device_Protection_No', 
            'Device_Protection_No_internet_service', 'Device_Protection_Yes', 'Tech_Support_No', 
            'Tech_Support_No_internet_service', 'Tech_Support_Yes', 'Streaming_TV_No', 
            'Streaming_TV_No_internet_service', 'Streaming_TV_Yes', 'Streaming_Movies_No', 
            'Streaming_Movies_No_internet_service', 'Streaming_Movies_Yes', 'Contract_Month-to-month', 
            'Contract_One_year', 'Contract_Two_year', 'Paperless_Billing_No', 'Paperless_Billing_Yes', 
            'Payment_Method_Bank_transfer_automatic', 'Payment_Method_Credit_card_automatic', 
            'Payment_Method_Electronic_check', 'Payment_Method_Mailed_check', 'Service_Count_0', 
            'Service_Count_1', 'Service_Count_2', 'Service_Count_3', 'Service_Count_4', 
            'Service_Count_5', 'Service_Count_6'
        ]}

        features['Tenure_Months'] = input_dict.get('Tenure_Months', 0)
        features['Monthly_Charges'] = input_dict.get('Monthly_Charges', 0)
        features['Total_Charges'] = input_dict.get('Total_Charges', 0)

        features['Gender_Male'] = input_dict['Gender_Male']
        features['Gender_Female'] = 1 if input_dict['Gender_Male'] == 0 else 0

        features['Senior_Citizen_Yes'] = input_dict['Senior_Citizen_Yes']
        features['Senior_Citizen_No'] = 1 if input_dict['Senior_Citizen_Yes'] == 0 else 0
        
        # Partner & Dependents
        features['Partner_Yes'] = input_dict['Partner_Yes']
        features['Partner_No'] = 1 if input_dict['Partner_Yes'] == 0 else 0
        features['Dependents_Yes'] = input_dict['Dependents_Yes']
        features['Dependents_No'] = 1 if input_dict['Dependents_Yes'] == 0 else 0

        if input_dict.get('Contract_Month_to_month') == 1:
            features['Contract_Month-to-month'] = 1
        elif input_dict.get('Contract_One_year') == 1:
            features['Contract_One_year'] = 1
        else:
            features['Contract_Two_year'] = 1

        features['Payment_Method_Electronic_check'] = input_dict['Payment_Method_Electronic_check']
        features['Paperless_Billing_Yes'] = input_dict['Paperless_Billing_Yes']
        features['Paperless_Billing_No'] = 1 if input_dict['Paperless_Billing_Yes'] == 0 else 0

        service_fields = [
            'Tech_Support_No', 'Online_Security_No', 'Online_Backup_No', 
            'Device_Protection_No', 'Streaming_TV_No', 'Streaming_Movies_No'
        ]
        no_count = sum(input_dict.get(field, 1) for field in service_fields)
        actual_service_count = 6 - no_count
        features[f'Service_Count_{actual_service_count}'] = 1

        # We pass only the values in the correct order
        raw_result = engine.predict_and_recommend([features])

        if not raw_result:
            return {"status": "error", "message": "Model returned no data."}

        prediction = raw_result[0]
        prob = prediction.get('probability', 0)
        risk_level = "High Risk" if prob > 0.80 else "Medium Risk" if prob > 0.50 else "Low Risk"
    
        return {
            "status": "success",
            "prediction": {
                "probability": f"{prob * 100:.2f}%",
                "risk": risk_level,
                "action": prediction.get('recommendation', 'N/A')
            }
        }

    except Exception as e:
        return {"error": str(e)}


       
# This allows you to run "python main.py" OR "uvicorn main:app --reload"
if __name__ == "__main__":
    print("--- STARTING UVICORN SERVER ---")
    # We use the app object directly here
    # Host '0.0.0.0' allows access from other devices on your network if needed
    uvicorn.run(app, host="127.0.0.1", port=8000)