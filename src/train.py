import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os

from src.preprocessing import load_and_preprocess

def train_model(data_path="data/synthetic_energy_data.csv", model_path="models/rf_model.pkl"):
    """
    Trains a Random Forest Regressor on the engineered tabular data.
    """
    print("Starting model training phase...")
    df = load_and_preprocess(data_path)
    
    # Define features (X) and target (y)
    X = df.drop(columns=['energy_kwh'])
    y = df['energy_kwh']
    
    # Time-series aware split (train on past, test on future).
    # Since it's synthetic we can use train_test_split, but test_size=0.2 mostly 
    # takes a chunk. For simplicity, standard split is fine.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    print(f"Training set: {X_train.shape[0]} records. Testing set: {X_test.shape[0]} records.")
    
    # Initialize and train the Random Forest
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    print("Model training complete.")
    
    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}.")
    return model, X_test, y_test

if __name__ == "__main__":
    train_model()
