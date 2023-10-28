import pandas as pd
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess the training data
data = pd.DataFrame()

for i in range(4):  # Assuming you have classified_data(0).csv to classified_data(3).csv
    file_name = f"D:\DL project\Classified data\\train\classified_data({i}).csv"
    data = data.append(pd.read_csv(file_name))

# Define features and target
X = data[['Acceleration_Type', 'Deceleration_Type']]
y = data['Good_Driver']

# Perform one-hot encoding to convert categorical data
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Create a Tkinter GUI
def import_data():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        test_data = pd.read_csv(file_path)

        # Preprocess the test data
        X_test_data = test_data[['Acceleration_Type', 'Deceleration_Type']]
        X_test_data = pd.get_dummies(X_test_data)

        # Make predictions
        predictions = classifier.predict(X_test_data)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        result_label.config(text=f"Accuracy: {accuracy * 100:.2f}%")

# Create a Tkinter window
window = tk.Tk()
window.title("Driver Evaluation")

# Create and configure the GUI components
import_button = tk.Button(window, text="Import Data", command=import_data)
import_button.pack(pady=10)
result_label = tk.Label(window, text="")
result_label.pack(pady=10)

# Start the GUI main loop
window.mainloop()
