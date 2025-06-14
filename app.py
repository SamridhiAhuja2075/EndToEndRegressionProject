import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from random import random

application=Flask(__name__)
app=application

#importing ridge regresssor and standard scaler 

ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))



@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = int(request.form.get('Classes'))
            Region = int(request.form.get('Region'))

            new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            result = ridge_model.predict(new_data_scaled)

            return render_template('home.html', result=result[0])
        
        except Exception as e:
            print("Error occurred:", e)  # Will show in terminal
            return render_template('home.html', result=f"Error: {str(e)}")  # Ensures valid response

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host='0.0.0.0')