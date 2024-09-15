import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Data Preprocessing
def load_and_preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Example preprocessing: remove missing values, and select relevant columns
    data = data.dropna()
    X = data[['area', 'bedrooms', 'bathrooms', 'floors']]  # Example features
    y = data['price']  # Target variable
    
    return X, y

# Model Training
def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the trained model to a file
    joblib.dump(model, 'house_price_model.pkl')
    
    print(f"Model trained with an R^2 score of {model.score(X_test, y_test)}")

# Load and train model (this would typically be triggered by some event)
if __name__ == "__main__":
    X, y = load_and_preprocess_data('house_prices.csv')
    train_model(X, y)

