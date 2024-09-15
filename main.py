# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # Import joblib for saving the model

# Load the dataset (ensure the dataset is in CSV format)
data = pd.read_csv('Housing.csv')

# Display the first few rows of the dataset
print(data.head())

# Data Preprocessing

# Convert categorical columns into numerical values
label_encoder = LabelEncoder()

# Columns with yes/no values
for column in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
    data[column] = label_encoder.fit_transform(data[column])

# Convert furnishingstatus into numerical values
data['furnishingstatus'] = label_encoder.fit_transform(data['furnishingstatus'])

# Features and target variable
X = data.drop('price', axis=1)  # Features
y = data['price']               # Target variable (Price)

# Normalize/Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training - Using Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Save the model to a file
joblib.dump(model, 'house_price_model.pkl')
print("Model saved as house_price_model.pkl")

# Save the scaler as well to ensure consistent scaling for future predictions
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as scaler.pkl")

# Predicting new input
def predict_price(input_features):
    # Convert the input features into a DataFrame with the same columns as the original data
    input_df = pd.DataFrame([input_features], columns=X.columns)
    
    # Load the scaler and scale the input features
    input_scaled = scaler.transform(input_df)
    
    # Predict the price
    predicted_price = model.predict(input_scaled)
    
    return predicted_price

# Example: Predict the price for a new house
new_house_features = [8000, 3, 2, 2, 1, 1, 0, 1, 1, 2, 1, 1]  # Example features
predicted_price = predict_price(new_house_features)
print(f'Predicted Price for the input features: {predicted_price[0]}')
