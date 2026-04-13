import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def predict_and_evaluate(model, X_test, y_test):
    """
    Generates predictions using the trained model and evaluates performance metrics.
    """
    print("Forecasting future test data...")
    predictions = model.predict(X_test)
    
    # Calculate error metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    # Compile results
    metrics = {
        "Mean Absolute Error (MAE)": round(mae, 2),
        "Root Mean Squared Error (RMSE)": round(rmse, 2),
        "R-squared (R2 Score)": round(r2, 4)
    }
    
    print("\n--- MODEL EVALUATION METRICS ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("--------------------------------\n")
    
    # Save metrics to text file
    with open("outputs/metrics.txt", "w") as f:
        f.write("--- MODEL EVALUATION METRICS ---\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
            
    # Return predictions for visualization
    return predictions

if __name__ == "__main__":
    with open("models/rf_model.pkl", "rb") as f:
        m = pickle.load(f)
    print("Please run main.py to pass X_test and y_test properly.")
