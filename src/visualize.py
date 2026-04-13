import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_visualizations(y_test, predictions, X_test, model):
    """
    Generates industry standard business plots.
    1. Actual vs Predicted line plot (shows tracking accuracy over time).
    2. Feature Importance bar plot (shows what drives energy consumption).
    """
    print("Generating visualizations...")
    os.makedirs("images", exist_ok=True)
    
    # 1. Actual vs Predicted Plot (Displaying a realistic subset e.g. 100 hours)
    plt.figure(figsize=(14, 6))
    subset_limit = 100
    plt.plot(y_test.values[:subset_limit], label="Actual Energy (kWh)", color='blue', linestyle='-', marker='.')
    plt.plot(predictions[:subset_limit], label="Predicted Energy (kWh)", color='orange', linestyle='--', alpha=0.8)
    
    plt.title("Actual vs. Predicted Energy Consumption (100 Hour Subset)", fontsize=14)
    plt.xlabel("Hours", fontsize=12)
    plt.ylabel("Energy (kWh)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/actual_vs_predicted.png", dpi=300)
    print("Saved -> images/actual_vs_predicted.png")
    
    # 2. Feature Importance
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    features = X_test.columns
    
    sns.barplot(x=importances, y=features, palette='viridis')
    plt.title("Feature Importance in Forecasting Model", fontsize=14)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("images/feature_importance.png", dpi=300)
    print("Saved -> images/feature_importance.png")
    print("All visualizations created successfully.")

if __name__ == "__main__":
    print("Please run main.py to generate visualizations.")
