# models.py
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def prepare_features(df):
    """Prepare features and target from dataframe"""
    # Define features and target
    X = df[['payload_mass_kg', 'orbit', 'launch_site', 'flight_number', 'reused']]
    y = df['landing_success']
    
    return X, y

def create_preprocessing_pipeline():
    """Create preprocessing pipeline for features"""
    numerical_features = ['payload_mass_kg', 'flight_number']
    categorical_features = ['orbit', 'launch_site', 'reused']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return preprocessor

def train_evaluate_models(X, y):
    """Train and evaluate multiple classification models"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Store results
        results[name] = {
            'pipeline': pipeline,
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    return X_test, y_test, results

def save_model(model, filename, folder='models'):
    """Save trained model to file"""
    # Create directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Save model
    file_path = os.path.join(folder, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {file_path}")

def load_model(filename, folder='models'):
    """Load trained model from file"""
    file_path = os.path.join(folder, filename)
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed_data/falcon9_processed.csv')
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Train and evaluate models
    X_test, y_test, results = train_evaluate_models(X, y)
    
    # Print results
    for name, result in results.items():
        print(f"\n{name} Results:")
        print(f"Accuracy: {result['classification_report']['accuracy']:.4f}")
        print(f"Precision: {result['classification_report']['1']['precision']:.4f}")
        print(f"Recall: {result['classification_report']['1']['recall']:.4f}")
        print(f"F1-Score: {result['classification_report']['1']['f1-score']:.4f}")
    
    # Save best model (assuming Random Forest is best)
    save_model(results['Random Forest']['pipeline'], 'random_forest_model.pkl')