# data_collection.py
import requests
import pandas as pd
import os

def fetch_spacex_data():
    """Fetch SpaceX launch data from the API"""
    spacex_url = "https://api.spacexdata.com/v4/launches/past"
    response = requests.get(spacex_url)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.json_normalize(data)
        return df
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

def save_data(df, filename, folder='data/raw_data'):
    """Save dataframe to file"""
    # Create directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Save data
    file_path = os.path.join(folder, filename)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    # Fetch and save SpaceX data
    spacex_df = fetch_spacex_data()
    if spacex_df is not None:
        save_data(spacex_df, 'spacex_launches_raw.csv')