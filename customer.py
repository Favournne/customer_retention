import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from xgboost import XGBClassifier
import joblib

# DATA & CLEANING
df = pd.read_excel(r"C:\Users\USER\customer_retention\Telco_customer_churn.xlsx", index_col="id")
df['Churn Reason'] = df['Churn Reason'].fillna('Active Customer')
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce').fillna(0)

#  FEATURE ENGINEERING (Our custom Service_Count logic)
services = ['Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies']
df['Service_Count'] = df[services].replace({'Yes': 1, 'No': 0, 'No internet service': 0}).sum(axis=1)

# PREPROCESSING
blacklist = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 
             'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 
             'Churn Value', 'Churn Reason', 'Churn Score', 'CLTV']

X = df.drop(columns=blacklist, errors='ignore')
X = pd.get_dummies(X)
y = df['Churn Value']

# XGBoost Column Formatting
X.columns = [c.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').replace('/', '_') for c in X.columns]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TRAINING THE SUPER MODEL
# Model A: XGBoost
xgb_model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, scale_pos_weight=3, random_state=42)
xgb_model.fit(X_train, y_train)

# Model B: Random Forest + SMOTE
pipeline = imbpipeline(steps=[
    ('smote', SMOTE(random_state=42)), 
    ('model', RandomForestClassifier(random_state=42))
])
param_grid = {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10, 20]}
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(X_train, y_train)

# The Ensemble (Soft Voting for Probabilities)
super_model = VotingClassifier(estimators=[('rf', grid_search.best_estimator_), ('xgb', xgb_model)], voting='soft')
super_model.fit(X_train, y_train)
joblib.dump(super_model, 'customer_churn_super_model_v1.pkl')

# GENERATING RESULTS
probs = super_model.predict_proba(X_test)[:, 1]
results = X_test.copy()
results['Churn_Probability'] = probs
results['Final_Verdict'] = (probs > 0.5).astype(int)

# RECOMMENDATION & WHAT-IF LOGIC
def get_recommendation(row):
    prob = row.get('Churn_Probability', 0)
    contract_m2m = row.get('Contract_Month-to-month', 0)
    tech_support_no = row.get('Tech_Support_No', 0)
    if prob > 0.80:
        level = 'High Risk'
        rec = "Offer 30% discount for 1 year contract migration" if contract_m2m == 1 else "Direct outreach: Loyalty manager personal call"
    elif prob > 0.50:
        level = 'Medium Risk'
        if tech_support_no == 1:
            rec = "Bundle free Tech Support for 6 months"
        elif contract_m2m == 1:
            rec = "Promote 'Cyber_Safe' bundle with Online Security."
        else:
            rec = "General loyalty discount coupon"
    else:
        level, rec = 'Low Risk', "Send marketing communication"
    return pd.Series([level, rec])

results[['Risk_level', 'Strategy']] = results.apply(get_recommendation, axis=1)

def what_if_analysis(customer_index, change_dict):
    test_customer = X_test.iloc[[customer_index]].copy()
    orig = super_model.predict_proba(test_customer)[0, 1]
    for feat, val in change_dict.items():
        if feat in test_customer.columns: test_customer[feat] = val
    new_p = super_model.predict_proba(test_customer)[0, 1]
    

#  FINAL OUTPUTS
what_if_analysis(10, {'Contract_Month-to-month': 0, 'Contract_One_year': 1})


class ChurnPredictor:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def predict_and_recommend(self, customer_data):
        # Ensure the data is in a DataFrame with correct columns
        df_input = pd.DataFrame(customer_data)
        
        # Probability Logic
        prob = self.model.predict_proba(df_input)[0, 1]
        
        # Recommendation Logic
        contract_m2m = df_input.get('Contract_Month-to-month', pd.Series([0])).iloc[0] if 'Contract_Month-to-month' in df_input.columns else 0
        tech_support_no = df_input.get('Tech_Support_No', pd.Series([0])).iloc[0] if 'Tech_Support_No' in df_input.columns else 0
        if prob > 0.80:
            level = 'High Risk'
            strategy = "Offer 30% discount for 1 year contract migration" if contract_m2m == 1 else "Direct outreach: Loyalty manager personal call"
        elif prob > 0.50:
            level = 'Medium Risk'
            if tech_support_no == 1:
                strategy = "Bundle free Tech Support for 6 months"
            else:
                strategy = "General loyalty discount coupon"
        else:
            level, strategy = 'Low Risk', "Send marketing communication"
        
        return {
            "probability": round(prob, 4),
            "risk_level": level,
            "recommended_strategy": strategy
        }


engine = ChurnPredictor(model=super_model, features=list(X.columns))
joblib.dump(engine, 'retention_engine_v1.pkl')

print(X_train.columns.tolist())















