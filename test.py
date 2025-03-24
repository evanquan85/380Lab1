import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('data/time_EMG1_EMG2_Data.csv')


plt.plot(df['EMG1'])
# Show the plot
plt.show()

plt.plot(df['EMG2'])
plt.show()

