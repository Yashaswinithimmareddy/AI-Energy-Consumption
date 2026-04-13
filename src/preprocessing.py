import pandas as pd

def load_and_preprocess(filepath="data/synthetic_energy_data.csv"):
    """
    Loads raw time-series data and engineers features for ML modeling.
    """
    print("Loading and preprocessing data...")
    df = pd.read_csv(filepath)
    
    # Convert string timestamp to datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Feature Engineering (Extracting components out of the timestamp)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Cyclic encoding for hour and month (to tell the model that hour 23 is close to hour 0)
    import numpy as np
    df['hour_sin'] = np.sin(df['hour'] * (2. * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2. * np.pi / 24))
    df['month_sin'] = np.sin((df['month'] - 1) * (2. * np.pi / 12))
    df['month_cos'] = np.cos((df['month'] - 1) * (2. * np.pi / 12))
    
    # Drop original timestamp as ML models only understand numbers
    features_df = df.drop(columns=['timestamp'])
    
    print(f"Data shape after preprocessing: {features_df.shape}")
    return features_df

if __name__ == "__main__":
    df = load_and_preprocess()
    print(df.head())
