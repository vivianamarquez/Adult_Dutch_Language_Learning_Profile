from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

app = Flask(__name__)

# Load the preprocessor and model
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('xgboost_best_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Get form data
        input_data = {
            'AaA': float(request.form['AaA']),
            'LoR': float(request.form['LoR']),
            'Edu.day': float(request.form['Edu_day']),
            'Enroll': float(request.form['Enroll']),
            'L1_mod': request.form['L1_mod'],
            'L2_mod': request.form['L2_mod']
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the input data
        X_processed = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True) 