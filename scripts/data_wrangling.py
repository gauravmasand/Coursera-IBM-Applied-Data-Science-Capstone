# data_wrangling.py
import pandas as pd
import numpy as np
import os

def load_data(filepath):
    """Load data from CSV file"""
    return pd.read_csv(filepath)

def prepare_spacex_data(df):
    """Clean and prepare SpaceX launch data"""
    # Filter for Falcon 9 rockets only
    falcon9_df = df[df['rocket.name'] == 'Falcon 9']
    
    # Extract key features
    falcon9_df['launch_date'] = pd.to_datetime(falcon9_df['date_utc']).dt.date
    falcon9_df['launch_year'] = pd.to_datetime(falcon9_df['date_utc']).dt.year
    
    # Create success column (1 for success, 0 for failure)
    falcon9_df['landing_success'] = falcon9_df['cores.0.landing_success'].apply(
        lambda x: 1 if x == True else 0)
    
    # Extract launch site information
    falcon9_df['launch_site'] = falcon9_df['launchpad'].apply(
        lambda x: x.split('/')[-1] if isinstance(x, str) else x)
    
    # Handle missing values
    falcon9_df['payload_mass_kg'] = falcon9_df['payload_mass_kg'].fillna(
        falcon9_df['payload_mass_kg'].mean())
    
    # Create reused flag
    falcon9_df['reused'] = falcon9_df['cores.0.reused'].apply(
        lambda x: 1 if x == True else 0)
    
    return falcon9_df

def save_processed_data(df, filename, folder='data/processed_data'):
    """Save processed dataframe to file"""
    # Create directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Save data
    file_path = os.path.join(folder, filename)
    df.to_csv(file_path, index=False)
    print(f"Processed data saved to {file_path}")

if __name__ == "__main__":
    # Load raw data
    raw_data_path = 'data/raw_data/spacex_launches_raw.csv'
    raw_df = load_data(raw_data_path)
    
    # Process data
    processed_df = prepare_spacex_data(raw_df)
    
    # Save processed data
    save_processed_data(processed_df, 'falcon9_processed.csv')