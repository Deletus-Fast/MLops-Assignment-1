import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Data Preprocessing
def load_and_preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Convert categorical variables to binary (0 and 1) for simplicity
    data['mainroad'] = data['mainroad'].map({'yes': 1, 'no': 0})
    data['guestroom'] = data['guestroom'].map({'yes': 1, 'no': 0})
    data['basement'] = data['basement'].map({'yes': 1, 'no': 0})
    data['hotwaterheating'] = data['hotwaterheating'].map({'yes': 1, 'no': 0})
    data['airconditioning'] = data['airconditioning'].map({'yes': 1, 'no': 0})
    data['prefarea'] = data['prefarea'].map({'yes': 1, 'no': 0})
    
    # Furnishing status will be treated as dummy variables (one-hot encoding)
    data = pd.get_dummies(data, columns=['furnishingstatus'], drop_first=True)

    # Features and target variable
    X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 
              'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus_semi-furnished', 
              'furnishingstatus_unfurnished']]
    y = data['price']  # Target variable
    
    return X, y

# Model Training
def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the trained model to a file
    joblib.dump(model, 'house_price_model.pkl')
    
    # Print the model performance
    print(f"Model trained with an R^2 score of {model.score(X_test, y_test)}")

# Load and train model (this would typically be triggered by some event)
if __name__ == "__main__":
    X, y = load_and_preprocess_data('Housing.csv')
    train_model(X, y)
