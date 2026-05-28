Markdown
# 📊 End-to-End Customer Retention Engine

A production-grade Machine Learning architecture that predicts customer churn probabilities using an XGBoost/Random Forest pipeline and serves real-time retention strategies. 

The project features a **decoupled architecture**: a high-performance **FastAPI** backend deployed on **Render** handling model inference, and a responsive **Streamlit** user interface running locally for management and operational workflows.

---

## 🏗️ Architecture Overview

```text
  ┌─────────────────────────┐               ┌─────────────────────────┐
  │   Streamlit Frontend    │  REST API     │     FastAPI Backend     │
  │     (Local Client)      │ ────────────> │     (Render Cloud)      │
  │  Runs inputs via form   │ <──────────── │ Executes ML Inference   │
  └─────────────────────────┘   JSON Res    
  └─────────────────────────┘


1. Backend (Inference Engine): Built with FastAPI. It handles incoming customer feature dictionaries, dynamically calculates engineering metrics (like missing Total_Charges and service counts), applies the unpickled machine learning wrapper pipeline, and outputs risk categories with explicit actionable workflows.

2. Frontend (Operator UI): Built with Streamlit. Provides a responsive three-column dashboard for entering tenant attributes, automatically parsing input tokens to interface smoothly with the remote cloud API.📂

Project Repository StructurePlaintextcustomer_retention/
│
├── .streamlit/
│   └── secrets.toml          # Local infrastructure secrets (API URL mapping)
├── main.py                   # Production FastAPI Application
├── app.py                    # Streamlit Dashboard UI Application
├── retention_engine_v1.pkl   # Serialized Joblib model pipeline wrapper
├── requirements.txt          # Shared production dependencies
└── README.md                 # Project documentation

⚙️ Core Business Logic & Decision TiersThe backend executes a custom ChurnPredictor namespace class that maps continuous prediction probabilities directly into automated customer success playbooks:

Probability Range,Calculated Risk Level,Operational Mitigation Strategy
> 70%,🔴 High Risk,Urgent: High-priority retention call & 50% discount offer
40% - 70%,🟡 Medium Risk,Offer 20% discount & personalized feedback survey
< 40%,🟢 Low Risk,Standard follow-up: Send newsletter & feature updates


🚀 Deployment & Installation1. Prerequisites & Shared DependenciesEnsure you have Python 3.9+ installed. Install the explicit production requirements locally or within your virtual environment:

pip install -r requirements.txt

2. Backend Cloud Setup (Render)

1. Commit your codebase to a GitHub repository, ensuring main.py, requirements.txt, and retention_engine_v1.pkl are all sitting in the root directory.

2. Log into the Render Dashboard and provision a new Web Service.

3. Link your repository and set the following build properties:
Build Command: pip install -r requirements.txt
Start Command: python -m uvicorn main:app --host 0.0.0.0 --port $PORT

4. Deploy the service and record the generated cloud domain string (e.g., https://your-api-service.onrender.com).

3. Local Frontend Setup

To link your local interactive environment to the remote cloud engine without hardcoding links into codebase lines:
1. Create a local secrets configuration path: .streamlit/secrets.toml

2. Add your live Render base endpoint to the environment configuration file:
Ini, TOML  
 # .streamlit/secrets.toml
   API_URL = "[https://your-api-service.onrender.com](https://your-api-service.onrender.com)"

💻 Running the ApplicationSince the heavy lifting and inference engine live permanently in the cloud on Render, you do not need to run backend servers on your local machine.To spin up the system for generation workflows, open your terminal and run the frontend:

streamlit run app.py

Navigate to http://localhost:8501 in your browser. The frontend will process input values, securely communicate across the internet to your cloud backend, and render real-time customer health metrics instantly.

⚠️ Note on Free-Tier Resource Allocation: This backend runs on an isolated cloud instance. If the service experiences 15 minutes of inactivity, Render puts the server into an automated sleep cycle. The very first prediction request of a session may experience a 40-50 second delay while the container reallocates memory and unpickles the ML structural weights. Subsequent inferences execute immediately.