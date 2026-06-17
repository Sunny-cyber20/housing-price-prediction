
"""
Housing Price Prediction Script
Standalone script to make predictions using the trained model
"""

import pandas as pd
import numpy as np
import pickle

def load_model_and_scaler():
    """Load trained model and scaler"""
    with open('models/best_model_tuned.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load feature columns
    train_data = pd.read_csv('data/X_train_scaled.csv')
    feature_columns = train_data.columns.tolist()

    return model, scaler, feature_columns

def predict_price(area, bedrooms, amenities_dict):
    """
    Predict house price

    Parameters:
    -----------
    area : float
        Area in sqft
    bedrooms : int
        Number of bedrooms
    amenities_dict : dict
        Dictionary of amenities (1 for yes, 0 for no)

    Returns:
    --------
    predicted_price : float
        Predicted price in rupees
    """

    model, scaler, feature_columns = load_model_and_scaler()

    # Create feature dictionary
    property_data = {
        'Area': area,
        'No. of Bedrooms': bedrooms,
    }

    # Add amenities
    property_data.update(amenities_dict)

    # Create dataframe
    data_df = pd.DataFrame([property_data])

    # Ensure all features are present
    for col in feature_columns:
        if col not in data_df.columns:
            data_df[col] = 0

    # Select required features
    data_df = data_df[feature_columns]

    # Scale
    data_scaled = scaler.transform(data_df)

    # Predict
    prediction = model.predict(data_scaled)[0]

    return prediction

if __name__ == "__main__":
    # Example usage
    print("Housing Price Prediction Script")
    print("=" * 50)

    # Example property
    area = 1500
    bedrooms = 2
    amenities = {
        'Resale': 0,
        'MaintenanceStaff': 1,
        'Gymnasium': 1,
        'SwimmingPool': 1,
        'AC': 1,
    }

    price = predict_price(area, bedrooms, amenities)

    print(f"Property Details:")
    print(f"  Area: {area} sqft")
    print(f"  Bedrooms: {bedrooms}")
    print(f"\nPredicted Price: ₹{price:,.0f}")
