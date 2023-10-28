import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load and preprocess the data from four CSV files
data_list = []

for csv_file in ['D:\DL project\Classified data\\train\classified_data(0).csv', 'D:\DL project\Classified data\\train\classified_data(1).csv', 'D:\DL project\Classified data\\train\classified_data(2).csv', 'D:\DL project\Classified data\\train\classified_data(3).csv']:
    data = pd.read_csv(csv_file)
    data = data[['time_seconds', 'speed_meters_per_second', 'acceleration_meters_per_second_squared', 'Good_Driver']]
    data['Good_Driver'] = data['Good_Driver'].astype(int)
    data_list.append(data)

# Concatenate the data from all files
all_data = pd.concat(data_list, ignore_index=True)

# Split the data into training and testing sets
X = all_data.drop('Good_Driver', axis=1)
y = all_data['Good_Driver']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save the trained model to a file
joblib.dump(model, 'driver_model.pkl')

print(f'Model trained and saved with accuracy: {accuracy}')
