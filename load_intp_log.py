import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

intp_filename = r'\\169.254.186.88\IntpLog\sine_full_joints_speed_100.txt'

# Load the .txt file and specify that it's in CSV format
df = pd.read_csv(intp_filename, delimiter=',', header=None)

# Select columns 6 to 11 (which are columns 5 to 10 in zero-based index)
selected_columns = df.iloc[:, 7:13]


# Create a figure and axes for 6 subplots
fig, axs = plt.subplots(6, 1, figsize=(10, 15))

# Loop through each column and plot in a separate subplot
for i, col in enumerate(selected_columns.columns):
    axs[i].plot(selected_columns[col])
    axs[i].set_title(f'Column {col+1}')
    axs[i].set_xlabel('Index')
    axs[i].set_ylabel('Value')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

