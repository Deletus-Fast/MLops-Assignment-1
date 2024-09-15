from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('house_price_model.pkl')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Parse incoming data (expecting JSON)
    data = request.json
    
    # Extract the features from the request
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
        1 if data.get('furnishingstatus', 'unfurnished') == 'furnished' else 0
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return the prediction as JSON
    return jsonify({'predicted_price': prediction[0]})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
