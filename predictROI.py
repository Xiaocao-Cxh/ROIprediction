import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Sample Input Data (you can create or assume a real dataset)
data = {
    'budget': [1000, 2000, 1500, 3000],
    'clicks': [100, 150, 120, 170],
    'cpc': [2.0, 1.8, 2.2, 1.7],
    'conversions': [20, 30, 25, 35],
    'platform': [0, 1, 0, 1],  # 0 for META, 1 for Google
    'campaign_duration': [10, 20, 15, 25],
    'roi': [200, 300, 250, 350]  # Target variable
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df.drop('roi', axis=1)
y = df['roi']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a RandomForest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Define column names for new data
columns = ['budget', 'clicks', 'cpc', 'conversions', 'platform', 'campaign_duration']

# Create a DataFrame for the new campaign
new_campaign = pd.DataFrame([[2500, 130, 1.9, 28, 1, 18]], columns=columns)

# Make predictions of ROI for the new campaign
predicted_roi = model.predict(new_campaign)
print(f"Predicted ROI for new campaign: {predicted_roi[0]}")
