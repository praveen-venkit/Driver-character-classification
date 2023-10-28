import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load your CSV data into a Pandas DataFrame
data = pd.read_csv("D:\DL project\Classified data\processed_data(0).csv")

# Step 2: Calculate the slope of acceleration
data['Slope'] = data['acceleration_meters_per_second_squared'].diff()  # Calculate the difference
data['Slope'] = data['Slope'].fillna(0)  # Fill NaN values with 0

# Step 3: Categorize acceleration based on the slope
def categorize_acceleration(slope):
    if slope > 1:
        return "Rapid Acceleration"
    elif slope > 0.5:
        return "Smooth Acceleration"
    elif slope < -1:
        return "Rapid Deceleration"
    elif slope < -0.5:
        return "Smooth Deceleration"
    else:
        return "No Acceleration/Deceleration"

data['Acceleration_Category'] = data['Slope'].apply(categorize_acceleration)

# Step 4: Create the 'Good_driver' column
data['Good_driver'] = np.where(data['Acceleration_Category'].str.contains("Smooth"), 1, 0)

# Step 5: Split the data into training and testing sets
X = data[['time_seconds', 'speed_meters_per_second', 'acceleration_meters_per_second_squared']]
y = data['Acceleration_Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train a feedforward neural network (Multi-Layer Perceptron) using Scikit-Learn
clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# Step 8: Predict acceleration categories on the test set
y_pred = clf.predict(X_test)

# Step 9: Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Step 10: Save the processed data to a new CSV file
data.to_csv("classified.csv", index=False)
