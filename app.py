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
    
    # Extract the features from the request (area, bedrooms, bathrooms, floors)
    features = np.array([data['area'], data['bedrooms'], data['bathrooms'], data['floors']]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return the prediction as JSON
    return jsonify({'predicted_price': prediction[0]})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
