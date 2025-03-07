from flask import Flask, render_template, request, make_response
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import openai
import os
from config import OPENAI_API_KEY

app = Flask(__name__)

# Load the preprocessor and model
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('xgboost_best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

def get_recommendation(profile, predicted_score):
    """Generate a personalized recommendation based on the learner's profile."""
    if not openai.api_key or openai.api_key == "your-api-key-here":
        return "Please set your OpenAI API key in config.py to get personalized recommendations."

    prompt = f"""As a Dutch language learning expert, provide a concise, personalized recommendation for a learner with the following profile:
    - Age of Arrival: {profile['AaA']} years old
    - Length of Residence: {profile['LoR']} years
    - Education Level: {profile['Edu.day']}
    - Time Enrolled in Course: {profile['Enroll']} months
    - Native Language Group: {profile['L1_mod']}
    - Second Language Group: {profile['L2_mod']}
    - Predicted Proficiency Score: {predicted_score:.2f}

    Please provide specific learning strategies and resources that would be most effective for this learner's profile.
    Keep the response under 150 words and focus on actionable advice."""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Dutch language learning expert providing personalized recommendations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Error generating recommendation: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    recommendation = None
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
        
        # Get recommendation
        recommendation = get_recommendation(input_data, prediction)
    
    # Create response with no-cache headers
    response = make_response(render_template('index.html', prediction=prediction, recommendation=recommendation))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5001) 