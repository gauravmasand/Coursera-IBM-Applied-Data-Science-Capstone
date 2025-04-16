# visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_success_by_year_plot(df, save_path=None):
    """Create line plot of launch success rate by year"""
    plt.figure(figsize=(12, 6))
    
    yearly_success = df.groupby('launch_year')['landing_success'].mean()
    yearly_launches = df.groupby('launch_year')['landing_success'].count()
    
    ax = yearly_success.plot(kind='line', marker='o', color='blue')
    plt.title('Launch Success Rate by Year')
    plt.xlabel('Year')
    plt.ylabel('Success Rate')
    plt.grid(True)
    
    # Add data point labels
    for i, v in enumerate(yearly_success):
        year = yearly_success.index[i]
        count = yearly_launches[year]
        ax.annotate(f"{v:.2f} ({count})", 
                   (year, v),
                   xytext=(5, 5),
                   textcoords='offset points')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    return plt

def create_correlation_heatmap(df, columns, save_path=None):
    """Create correlation heatmap for specified columns"""
    plt.figure(figsize=(10, 8))
    
    corr_matrix = df[columns].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Heatmap saved to {save_path}")
    
    return plt

def create_bar_chart(data, x, y, title, save_path=None):
    """Create bar chart"""
    plt.figure(figsize=(10, 6))
    
    sns.barplot(data=data, x=x, y=y)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Bar chart saved to {save_path}")
    
    return plt

if __name__ == "__main__":
    # Test visualization functions with sample data
    df = pd.read_csv('data/processed_data/falcon9_processed.csv')
    
    # Create and save plots
    create_success_by_year_plot(df, 'images/visualizations/success_by_year.png')
    
    create_correlation_heatmap(df, 
                              ['payload_mass_kg', 'flight_number', 'landing_success', 'reused'],
                              'images/visualizations/correlation_matrix.png')