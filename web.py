# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 2025
@author: Hala
"""

from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)

# تحميل الموديل
model = load('diabetes_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # جلب القيم من الفورم وتحويلها لأرقام
        input_features = [float(x) for x in request.form.values()]
        input_array = np.array([input_features])

        prediction = model.predict(input_array)
        result = 'High Risk of Diabetes' if prediction[0] == 1 else 'Low Risk of Diabetes'

        return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == '__main__':
    app.run(debug=True)
