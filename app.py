from flask import Flask, render_template, request, jsonify

import pickle
import numpy as np


app = Flask(__name__)

@app.route('/')
def homepage():
	return render_template('index.html')
	
	
@app.route('/about')
def aboutpage():
	return render_template('about.html')
	
	
@app.route('/predict')
def predictpage():
	return render_template('predict.html')
	

@app.route('/explore')
def explorepage():
	return render_template('explore.html')
	

@app.route('/summary')
def summarypage():
	return render_template('summary.html')
	
	
@app.route('/process')
def processpage():
	return render_template('process.html')
	



def load_model():
    with open('saved_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


	
@app.route('/machineresponse', methods=['POST'])
def machineresponse():
    data = load_model()

    loaded_prediction_model  = data["model"]
    le_gender = data["le_gender"]
    le_senior_citizen = data["le_senior_citizen"]
    le_partner = data["le_partner"]
    le_dependents = data["le_dependents"]
    le_phone_service = data["le_phone_service"]
    le_multiple_lines = data["le_multiple_lines"]
    le_internet_service = data["le_internet_service"]
    le_online_security = data["le_online_security"]
    le_online_backup = data["le_online_backup"]
    le_device_protection = data["le_device_protection"]
    le_tech_support = data["le_tech_support"]
    le_streaming_tv = data["le_streaming_tv"]
    le_streaming_movies = data["le_streaming_movies"]
    le_contract = data["le_contract"]
    le_paperless_billing = data["le_paperless_billing"]
    le_payment_method = data["le_payment_method"]


    uploadeddata = request.json
    

    
    # Selectors for each variable
    gender = uploadeddata.get('gender')
    senior_citizen = uploadeddata.get('seniorcitizen')
    partner = uploadeddata.get('partner')
    dependents = uploadeddata.get('dependents')
    tenure_months = uploadeddata.get('tenuremonths')
    phone_service = uploadeddata.get('phoneservices')
    multiple_lines = uploadeddata.get('multiplelines')
    internet_service = uploadeddata.get('internetservice')
    online_security = uploadeddata.get('onlinesecurity')
    online_backup = uploadeddata.get('onlinebackup')
    device_protection = uploadeddata.get('deviceprotection')
    tech_support = uploadeddata.get('techsupport')
    streaming_tv = uploadeddata.get('streamingtv')
    streaming_movies = uploadeddata.get('streamingmovies')
    contract = uploadeddata.get('contract')
    paperless_billing = uploadeddata.get('paperlessbilling')
    payment_method = uploadeddata.get('paymentmethod')
    monthly_charges = uploadeddata.get('monthlycharges')
    total_charges = uploadeddata.get('totalcharges')
    cltv = uploadeddata.get('cltv')
    
    
    X = np.array([[
    gender, senior_citizen, partner, dependents,tenure_months, phone_service,
    multiple_lines, internet_service, online_security, online_backup,
    device_protection, tech_support, streaming_tv, streaming_movies,
    contract, paperless_billing, payment_method, monthly_charges,total_charges,cltv 
    ]])
    try :
        X[:, 0] = le_gender.transform(X[:, 0])
        X[:, 1] = le_senior_citizen.transform(X[:, 1])
        X[:, 2] = le_partner.transform(X[:, 2])
        X[:, 3] = le_dependents.transform(X[:, 3])
        X[:, 5] = le_phone_service.transform(X[:, 5])
        X[:, 6] = le_multiple_lines.transform(X[:, 6])
        X[:, 7] = le_internet_service.transform(X[:, 7])
        X[:, 8] = le_online_security.transform(X[:, 8])
        X[:, 9] = le_online_backup.transform(X[:, 9])
        X[:, 10] = le_device_protection.transform(X[:, 10])
        X[:, 11] = le_tech_support.transform(X[:, 11])
        X[:, 12] = le_streaming_tv.transform(X[:, 12])
        X[:, 13] = le_streaming_movies.transform(X[:, 13])
        X[:, 14] = le_contract.transform(X[:, 14])
        X[:, 15] = le_paperless_billing.transform(X[:, 15])
        X[:, 16] = le_payment_method.transform(X[:, 16])
        
    except Exception as error:
        return jsonify( {'status': 'error', 'message' : str(error) })
        
    X = X.astype(float)

    churn_prediction = loaded_prediction_model.predict(X)
    if churn_prediction[0] == 0:
        return jsonify( { 'status': 'success', 'message' : "This customer is likely to stay with the company" } )
    elif churn_prediction[0] == 1:
        return jsonify( { 'status': 'success', 'message' : "This customer is likely to leave the company" } )
    else:
        return jsonify( { 'status': 'error', 'message' : "Unexpected prediction" } )

  

  
  
if __name__ == '__main__':
	app.run(host='127.0.0.1', port=8080, debug=True)


