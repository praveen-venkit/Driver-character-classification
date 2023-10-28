import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the CSV file
data = pd.read_csv('D:\DL project\Classified data\\train\classified_data(1).csv')

# Select relevant columns
data = data[['time_seconds', 'speed_meters_per_second', 'acceleration_meters_per_second_squared', 'Good_Driver']]

# Encode the 'Good_Driver' column
data['Good_Driver'] = data['Good_Driver'].astype(int)

# Split the data into training and testing sets
X = data.drop('Good_Driver', axis=1)
y = data['Good_Driver']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the percentage of good driving (1) and bad driving (0)
percentage_good_driving = (y_pred.sum() / len(y_pred)) * 100
percentage_bad_driving = 100 - percentage_good_driving

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(f'Percentage of Good Driving: {percentage_good_driving}%')
print(f'Percentage of Bad Driving: {percentage_bad_driving}%')
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(confusion)
