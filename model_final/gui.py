import tkinter as tk
from tkinter import filedialog
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the pre-trained model
model = joblib.load('D:\DL project\model_final\driver_model.pkl')

def calculate_percentage():
    file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    if file_path:
        data = pd.read_csv(file_path)
        # Preprocess the data (same preprocessing as in the training code)
        data = data[['time_seconds', 'speed_meters_per_second', 'acceleration_meters_per_second_squared']]
        # Make predictions
        percentages = model.predict_proba(data)[:, 1] * 100
        good_percentage.set(f'Percentage of Good Driving: {percentages.mean():.2f}%')
        bad_percentage.set(f'Percentage of Bad Driving: {100 - percentages.mean():.2f}%')

        # Create and display pie chart
        plt.figure(figsize=(5, 5))
        labels = ['Good Driving', 'Bad Driving']
        sizes = [percentages.mean(), 100 - percentages.mean()]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title('Driving Classification')

        canvas = FigureCanvasTkAgg(plt.gcf(), master=app)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()

app = tk.Tk()
app.title('Driver Classification')

good_percentage = tk.StringVar()
bad_percentage = tk.StringVar()

calculate_button = tk.Button(app, text='Calculate Percentage', command=calculate_percentage)
good_label = tk.Label(app, textvariable=good_percentage)
bad_label = tk.Label(app, textvariable=bad_percentage)

calculate_button.pack()
good_label.pack()
bad_label.pack()

app.mainloop()
