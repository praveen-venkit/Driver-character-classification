import pandas as pd
import os

# Define a function to classify acceleration
def classify_acceleration(acceleration, max_positive_accel, min_positive_accel):
    if acceleration == 0:
        return "Stationary"
    elif acceleration > 0:
        if acceleration > (max_positive_accel - min_positive_accel)//2:
            return "Rapid Acceleration"
        else:
            return "Smooth Acceleration"
    else:
        return None

# Define a function to classify deceleration
def classify_deceleration(acceleration, max_negative_accel, min_negative_accel):
    if acceleration == 0:
        return "Stationary"
    elif acceleration < 0:
        if acceleration < (min_negative_accel - max_negative_accel)//2:
            return "Rapid Deceleration"
        else:
            return "Smooth Deceleration"
    else:
        return None

# Process multiple files
for i in range(4):
    input_file = f"D:\DL project\processed data\processed_data({i}).csv"
    output_file = f"D:\DL project\Classified data\\train\classified_data({i}).csv"

    # Read the CSV file
    data = pd.read_csv(input_file)

    # Calculate max positive acceleration and min negative acceleration
    max_positive_accel = data[data['acceleration_meters_per_second_squared'] > 0]['acceleration_meters_per_second_squared'].max()
    min_positive_accel = data[data['acceleration_meters_per_second_squared'] > 0]['acceleration_meters_per_second_squared'].min()
    max_negative_accel = data[data['acceleration_meters_per_second_squared'] < 0]['acceleration_meters_per_second_squared'].max()
    min_negative_accel = data[data['acceleration_meters_per_second_squared'] < 0]['acceleration_meters_per_second_squared'].min()

    # Apply classification functions and create new columns
    data['Acceleration_Type'] = data['acceleration_meters_per_second_squared'].apply(
        lambda x: classify_acceleration(x, max_positive_accel, min_positive_accel)
    )
    data['Deceleration_Type'] = data['acceleration_meters_per_second_squared'].apply(
        lambda x: classify_deceleration(x, max_negative_accel, min_negative_accel)
    )

    # Create a 'Good_Driver' column based on smoothness criteria
# Create a 'Good_Driver' column based on smoothness criteria (using OR)
    data['Good_Driver'] = (data['Acceleration_Type'] == 'Smooth Acceleration') | (data['Deceleration_Type'] == 'Smooth Deceleration')
    #data['Good_Driver'] = data['Good_Driver'].astype(int)


    # Write the updated data to a new CSV file
    data.to_csv(output_file, index=False)

    print(f"Classification for {input_file} completed and saved to {output_file}")
