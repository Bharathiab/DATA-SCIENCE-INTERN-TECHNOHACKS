import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:\\Users\\S.Bharathi\\Downloads\\temperature.csv\\temperature.csv'
df = pd.read_csv(file_path)

# Convert 'datetime' column to datetime format
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract date features
df['Year'] = df['datetime'].dt.year
df['Month'] = df['datetime'].dt.month
df['Day'] = df['datetime'].dt.day
df['Hour'] = df['datetime'].dt.hour
df['Weekday'] = df['datetime'].dt.weekday

# Get the list of city columns
city_columns = [col for col in df.columns if col != 'datetime']

# Create a figure for subplots
num_cities = len(city_columns)
cols = 3  # Number of columns in the subplot grid
rows = (num_cities + cols - 1) // cols  # Calculate number of rows needed

plt.figure(figsize=(15, 5 * rows))

for i, city in enumerate(city_columns):
    print(f"Processing city: {city}")
    
    # Prepare features and target
    X = df[['Year', 'Month', 'Day', 'Hour', 'Weekday']]
    y = df[city]  # Temperature for the current city
    
    # Add a constant to the features for the intercept term
    X = sm.add_constant(X)
    
    # Split the data into training and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train a Linear Regression model
    model = sm.OLS(y_train, X_train)
    results = model.fit()
    
    # Make predictions
    predictions = results.predict(X_test)
    
    # Evaluate the model
    residuals = y_test - predictions
    mse = np.mean(residuals**2)
    print(f'Mean Squared Error for {city}: {mse}')
    
    # Create a subplot for each city
    plt.subplot(rows, cols, i + 1)
    plt.plot(y_test.reset_index(drop=True), label='Actual', color='blue', linestyle='--')
    plt.plot(predictions.reset_index(drop=True), label='Predicted', color='red')
    plt.title(f'{city} (MSE: {mse:.2f})')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
