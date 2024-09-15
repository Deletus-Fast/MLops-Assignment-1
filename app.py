from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Parse incoming data (expecting JSON)
    data = request.json
    
    # Extract the features from the request and convert them to match the training data
    features = np.array([
        data.get('area', 0),
        data.get('bedrooms', 0),
        data.get('bathrooms', 0),
        data.get('stories', 0),
        1 if data.get('mainroad', 'no') == 'yes' else 0,
        1 if data.get('guestroom', 'no') == 'yes' else 0,
        1 if data.get('basement', 'no') == 'yes' else 0,
        1 if data.get('hotwaterheating', 'no') == 'yes' else 0,
        1 if data.get('airconditioning', 'no') == 'yes' else 0,
        data.get('parking', 0),
        1 if data.get('prefarea', 'no') == 'yes' else 0,
        2 if data.get('furnishingstatus', 'unfurnished') == 'furnished' 
        else 1 if data.get('furnishingstatus', 'unfurnished') == 'semi-furnished' else 0
    ]).reshape(1, -1)
    
    # Convert features to a DataFrame to retain column names and structure
    columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
               'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea',
               'furnishingstatus']
    
    input_df = pd.DataFrame(features, columns=columns)
    
    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    
    # Return the prediction as JSON
    return jsonify({'predicted_price': prediction[0]})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
