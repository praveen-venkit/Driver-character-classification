import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
input_file = 'dataset_gps.csv'
df = pd.read_csv(input_file)

# Convert the Unix timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# Sort the DataFrame by timestamp
df.sort_values(by='timestamp', inplace=True)

# Calculate time in seconds relative to the first timestamp (start from 0 seconds)
df['time_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

# Calculate acceleration (change in speed over time)
df['acceleration_meters_per_second_squared'] = df['speed_meters_per_second'].diff() / df['time_seconds'].diff()

# Create a new DataFrame with time in seconds, speed, and acceleration
processed_df = df[['time_seconds', 'speed_meters_per_second', 'acceleration_meters_per_second_squared']]

# Save the processed data to a new CSV file
output_file = 'processed_data.csv'
processed_df.to_csv(output_file, index=False)

print(f'Processed data saved to {output_file}')



# Extract data for the graph
time_seconds = df['time_seconds']
speed = df['speed_meters_per_second']
acceleration = df['acceleration_meters_per_second_squared']

# Create a line graph
plt.figure(figsize=(10, 6))
plt.plot(time_seconds, speed, label='Speed (m/s)', linewidth=2)
plt.plot(time_seconds, acceleration, label='Acceleration (m/s^2)', linewidth=2)
plt.xlabel('Time (seconds)')
plt.ylabel('Speed/Acceleration')
plt.title('Time vs Speed vs Acceleration')
plt.legend()
plt.grid(True)

# Show the graph or save it to a file
plt.show()