import pandas as pd
import numpy as np

def generate_data(filepath="data/synthetic_energy_data.csv", days=365):
    """
    Generates synthetic hourly energy consumption data for a factory.
    Factors included: Time of day, day of week, temperature, and baseline load.
    """
    print(f"Generating {days} days of synthetic energy data...")
    
    # Create hourly timestamps
    date_rng = pd.date_range(start='2023-01-01', end=pd.to_datetime('2023-01-01') + pd.Timedelta(days=days), freq='h')
    df = pd.DataFrame(date_rng, columns=['timestamp'])
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Simulate Temperature (fluctuates daily and monthly)
    # Base temp varies by month (Jan cold, Jul hot)
    month_temp_base = {1: 10, 2: 12, 3: 15, 4: 20, 5: 25, 6: 30, 7: 32, 8: 31, 9: 27, 10: 20, 11: 15, 12: 11}
    df['base_temp'] = df['month'].map(month_temp_base)
    # Add daily temp variance (colder at night, hotter at 2 PM)
    df['temperature_c'] = df['base_temp'] + 5 * np.sin((df['hour'] - 8) * (np.pi / 12)) + np.random.normal(0, 1.5, len(df))
    
    # Simulate Energy Consumption (kWh)
    base_load = 500  # constant minimal factory load
    
    # 1. Day time load (factory operations 8 AM to 6 PM)
    df['operation_load'] = np.where((df['hour'] >= 8) & (df['hour'] <= 18), 1000, 200)
    
    # 2. Weekend reduction (factory mostly closed on Saturday=5, Sunday=6)
    df['weekend_penalty'] = np.where(df['day_of_week'] >= 5, 0.4, 1.0)
    
    # 3. Temperature effect (Cooling/AC load when hot)
    df['hvac_load'] = np.where(df['temperature_c'] > 22, (df['temperature_c'] - 22) * 50, 0)
    
    # Combine to total energy
    df['energy_kwh'] = (base_load + df['operation_load'] * df['weekend_penalty'] + df['hvac_load'])
    # Add some random noise to make it realistic
    df['energy_kwh'] += np.random.normal(0, 50, len(df))
    
    # Ensure no negative energy
    df['energy_kwh'] = df['energy_kwh'].clip(lower=0)
    
    # Select final columns
    final_df = df[['timestamp', 'temperature_c', 'energy_kwh']]
    
    # Save to CSV
    final_df.to_csv(filepath, index=False)
    print(f"Data successfully generated and saved to {filepath}.")
    return final_df

if __name__ == "__main__":
    generate_data()
