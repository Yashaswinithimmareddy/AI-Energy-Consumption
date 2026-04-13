import os
from src.data_generator import generate_data
from src.train import train_model
from src.predict import predict_and_evaluate
from src.visualize import generate_visualizations

def run_pipeline():
    print("==============================================")
    print(" AI-POWERED ENERGY CONSUMPTION FORECASTING    ")
    print("==============================================\n")
    
    # Check if directories exist
    for dir_name in ['data', 'models', 'outputs', 'images']:
        os.makedirs(dir_name, exist_ok=True)
    
    # Phase 2: Dataset Loading and Generation
    generate_data("data/synthetic_energy_data.csv", days=365)
    
    # Phase 3, 4, 5: Clean, Feature Engineer, and Train Model
    model, X_test, y_test = train_model("data/synthetic_energy_data.csv", "models/rf_model.pkl")
    
    # Phase 6 & 7: Evaluate and Forecast
    predictions = predict_and_evaluate(model, X_test, y_test)
    
    # Phase 8: Visualize results
    generate_visualizations(y_test, predictions, X_test, model)
    
    print("\n==============================================")
    print(" PROJECT EXECUTION COMPLETED SUCCESSFULLY! ")
    print(" Check 'outputs/' and 'images/' folders.")
    print("==============================================")

if __name__ == "__main__":
    run_pipeline()
