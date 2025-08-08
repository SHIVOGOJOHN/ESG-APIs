from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load models and scalers
esg_fin = joblib.load('Models-pkl/esg_fin_returns.pkl')
esg_scaler_fin = joblib.load('Models-pkl/esg_scaler_fin.pkl')

esg_profit = joblib.load('Models-pkl/esg_profit_margin.pkl')
esg_scaler_prft= joblib.load('Models-pkl/esg_scaler_prft.pkl')

esg_roe = joblib.load('Models-pkl/esg_roe.pkl')
esg_scaler_roe= joblib.load('Models-pkl/esg_scaler_roe.pkl')

env_fin = joblib.load('Models-pkl/env_fin_returns.pkl')
env_scaler_fin = joblib.load('Models-pkl/env_scaler_fin.pkl')

env_prft = joblib.load('Models-pkl/env_profit_margin.pkl')
env_scaler_prft = joblib.load('Models-pkl/env_scaler_prft.pkl')

env_roe = joblib.load('Models-pkl/env_roe.pkl')
env_scaler_roe = joblib.load('Models-pkl/env_scaler_roe.pkl')

gov_fin = joblib.load('Models-pkl/gov_fin_returns.pkl')
gov_scaler_fin = joblib.load('Models-pkl/gov_scaler_fin.pkl')

gov_prft = joblib.load('Models-pkl/gov_profit_margin.pkl')
gov_scaler_prft = joblib.load('Models-pkl/gov_scaler_prft.pkl')

gov_roe = joblib.load('Models-pkl/gov_roe.pkl')
gov_scaler_roe = joblib.load('Models-pkl/gov_scaler_roe.pkl')

soc_fin = joblib.load('Models-pkl/soc_fin_returns.pkl')
soc_scaler_fin = joblib.load('Models-pkl/soc_scaler_fin.pkl')

soc_prft = joblib.load('Models-pkl/soc_profit_margin.pkl')
soc_scaler_prft = joblib.load('Models-pkl/soc_scaler_prft.pkl')

soc_roe = joblib.load('Models-pkl/soc_roe.pkl')
soc_scaler_roe = joblib.load('Models-pkl/soc_scaler_roe.pkl')


# Define pydantic scheme per model
class EsgInput(BaseModel):
    carbon_emissions: float
    renewable_energy_ratio: float
    waste_management_score: float
    water_usage_intensity: float
    biodiversity_score: float
    excluded_sector: float
    employee_satisfaction: float
    diversity_index: float
    turnover_rate: float
    philanthropy_spend: float
    supply_chain_ethics_score: float
    data_privacy_compliance: float
    independent_board_ratio: float
    executive_pay_ratio: float
    proxy_voting_score: float
    risk_mgmt_score: float
    audit_transparency: float
    governance_framework_score: float
    esg_score: float

class EnvInput(BaseModel):
    carbon_emissions: float
    renewable_energy_ratio: float
    waste_management_score: float
    water_usage_intensity: float
    biodiversity_score: float
    esg_score: float

class SocInput(BaseModel):
    employee_satisfaction: float
    diversity_index: float
    turnover_rate: float
    philanthropy_spend: float
    supply_chain_ethics_score: float
    data_privacy_compliance: float
    esg_score: float

class GovInput(BaseModel):
    independent_board_ratio: float
    executive_pay_ratio: float
    proxy_voting_score: float
    risk_mgmt_score: float
    audit_transparency: float
    governance_framework_score: float
    esg_score: float

expected_order = ['carbon_emissions', 'renewable_energy_ratio', 'waste_management_score',
        'water_usage_intensity', 'biodiversity_score', 'excluded_sector',
        'employee_satisfaction', 'diversity_index', 'turnover_rate',
        'philanthropy_spend', 'supply_chain_ethics_score',
        'data_privacy_compliance', 'independent_board_ratio',
        'executive_pay_ratio', 'proxy_voting_score', 'risk_mgmt_score',
        'audit_transparency', 'governance_framework_score', 'esg_score']
    
env_order = ['carbon_emissions',
        'renewable_energy_ratio',
        'waste_management_score',
        'water_usage_intensity',
        'biodiversity_score',
        'esg_score']

soc_order = ['employee_satisfaction',
        'diversity_index',
        'turnover_rate',
        'philanthropy_spend',
        'supply_chain_ethics_score',
        'data_privacy_compliance',
        'esg_score']

gov_order = ['independent_board_ratio',
        'executive_pay_ratio',
        'proxy_voting_score',
        'risk_mgmt_score',
        'audit_transparency',
        'governance_framework_score',
        'esg_score']

# Testing route
@app.get("/ping")
def ping():
    return {"message": "pong"}
    
# Route for combined esg financial prediction
@app.post('/esg_financial_returns')
def predict_esg(input: EsgInput):
    input_dict = input.dict()
    input_array = np.array([[input_dict[feature] for feature in expected_order]])
    input_scaled = esg_scaler_fin.transform(input_array)
    prediction = esg_fin.predict(input_scaled)
    return {"prediction": float(prediction[0])}

# Route for combined esg- return on equity- pediction
@app.post('/esg_roe')
def predict_esg_roe(input: EsgInput):
    input_dict = input.dict()
    input_array = np.array([[input_dict[feature] for feature in expected_order]])
    input_scaled = esg_scaler_roe.transform(input_array)
    prediction = esg_roe.predict(input_scaled)
    return {"prediction": float(prediction[0])}

# Route for combined esg profit margin
@app.post('/esg_profit_margin')
def predict_esg_profit(input: EsgInput):
    input_dict = input.dict()
    input_array = np.array([[input_dict[feature] for feature in expected_order]])
    input_scaled = esg_scaler_prft.transform(input_array)
    prediction = esg_profit.predict(input_scaled)
    return {"prediction": float(prediction[0])}

# Route for env-financial return
@app.post('/env_financial_return')
def predict_env_fin(input: EnvInput):
    input_dict = input.dict()
    input_array = np.array([[input_dict[feature] for feature in env_order]])
    input_scaled =env_scaler_fin.transform(input_array)
    prediction = env_fin.predict(input_scaled)
    return {"prediction": float(prediction[0])}

# Route for env-roe
@app.post('/env_roe')
def predict_env_roe(input: EnvInput):
    input_dict = input.dict()
    input_array = np.array([[input_dict[feature] for feature in env_order]])
    input_scaled = env_scaler_roe.transform(input_array)
    prediction = env_roe.predict(input_scaled)
    return {"prediction": float(prediction[0])}

# Route for env-profit margins#
@app.post('/env_prft')
def predict_env_prft(input: EnvInput):
    input_dict = input.dict()
    input_array = np.array([[input_dict[feature] for feature in env_order]])
    input_scaled = env_scaler_prft.transform(input_array)
    prediction = env_prft.predict(input_scaled)
    return {"prediction": float(prediction[0])}

# Route for soc-financial returns
@app.post('/soc_financial_return')
def predict_env_fin(input: SocInput):
    input_dict = input.dict()
    input_array = np.array([[input_dict[feature] for feature in soc_order]])
    input_scaled =soc_scaler_fin.transform(input_array)
    prediction = soc_fin.predict(input_scaled)
    return {"prediction": float(prediction[0])}

# Route for soc-roe
@app.post('/soc_roe')
def predict_env_fin(input: SocInput):
    input_dict = input.dict()
    input_array = np.array([[input_dict[feature] for feature in soc_order]])
    input_scaled =soc_scaler_roe.transform(input_array)
    prediction = soc_roe.predict(input_scaled)
    return {"prediction": float(prediction[0])}

# Route for soc-profit margin
@app.post('/soc_prft')
def predict_env_fin(input: SocInput):
    input_dict = input.dict()
    input_array = np.array([[input_dict[feature] for feature in soc_order]])
    input_scaled =soc_scaler_prft.transform(input_array)
    prediction = soc_prft.predict(input_scaled)
    return {"prediction": float(prediction[0])}

# Route for gov-financial returns
@app.post('/gov_fin')
def predict_gov_fin(input: GovInput):
    input_dict = input.dict()
    input_array = np.array([[input_dict[feature] for feature in gov_order]])
    input_scaled = gov_scaler_fin.transform(input_array)
    prediction = gov_fin.predict(input_scaled)
    return {"prediction": float(prediction[0])}

# Route for gov-roe
@app.post('/gov_roe')
def predict_gov_roe(input: GovInput):
    input_dict = input.dict()
    input_array = np.array([[input_dict[feature] for feature in gov_order]])
    input_scaled = gov_scaler_roe.transform(input_array)
    prediction = gov_roe.predict(input_scaled)
    return {"pediction": float(prediction[0])}

# Route for gov-profit margins
@app.post('/gov_prft')
def predict_gov_prft(input: GovInput):
    input_dict = input.dict()
    input_array = np.array([[input_dict[feature] for feature in gov_order]])
    input_scaled = gov_scaler_prft.transform(input_array)
    prediction = gov_prft.predict(input_scaled)
    return {"prediction": float(prediction[0])}
