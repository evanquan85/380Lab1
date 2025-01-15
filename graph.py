"""
0.5 msec/sample	Sample rate =	2000Hz
2 channels
EMG (5 - 250 Hz w/notch)
mV
Clench Force
kg
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
# read CSV
df = pd.read_csv('B01_Fatigue.csv')
df.columns = ['Time', 'EMG', 'Force']
sampling_rate=2000
time_basic = df.index/sampling_rate
# EMG & Force offset. If offset < 0.1, ignore it
def calculate_offsets(df):
    emg_mean = df['EMG'].iloc[:10000].mean()
    force_mean = df['Force'].iloc[:10000].mean()
    if abs(emg_mean) >= 0.1:
        df['EMG'] = df['EMG'] - emg_mean
    if abs(force_mean) >= 0.1:
        df['Force']= df['Force'] - force_mean
    return df
df = calculate_offsets(df)

#Full wave rectify signal
df['EMG'] = df['EMG'].abs()
df['Force'] = df['Force'].abs()
#Smooth and root-mean-square EMG
window_size = 50
df['EMG_smoothed'] = df['EMG'].rolling(window=window_size, center=True).mean()
def calculate_rms(signal):
    return np.sqrt(np.mean(signal**2))
df['EMG_rms'] = df['EMG'].rolling(window=1000).apply(calculate_rms, raw=True)
# Create a figure and axes object """

fig, ax = plt.subplots()
# Plot the EMG column

ax.plot(df['EMG_rms'], label='EMG (mV)')

# Set the x-axis limits
ax.set_ylim(0, 0.4)
ax.set_xlim(78000, 84000)

# Set labels and title
ax.set_xlabel('Index Rate')
ax.set_ylabel('EMG (mV)')
ax.set_title('EMG Signal')

# Add a legend
ax.legend()

# Display the plot
plt.show()
# 4 Increasing contractions
# Determine average rectified EMG for each contraction
# start contract_1 = 45500 | end = 47750 | length = 2250 (just over 1s)
# start contract_2 = 54500 | end = 56750 | length = 2250
# start contract_3 = 69100 | end = 71350 | length = 2250
# start contract_4 = 78800 | end = 81050 | length = 2250
contractions = [
    {'start': 45500, 'end': 47750, 'title': '25% EMG signal'},  # Contraction 1
    {'start': 54500, 'end': 56750, 'title': '50% EMG signal'},  # Contraction 2
    {'start': 69100, 'end': 71350,'title': '75% EMG signal'}, # Contraction 3
    {'start': 78800, 'end': 81050, 'title': '100% EMG signal'} # Contraction 4
]
def calculate_aremg(df, contractions):
    aremg_values = []

    for contraction in contractions:
        segment = df['EMG'].iloc[contraction['start']:contraction['end']]
        rectified_segment = np.abs(segment)
        aremg = rectified_segment.mean()
        aremg_values.append(aremg)
    return aremg_values

# Calculate AREM for each contraction
aremg_values = calculate_aremg(df, contractions)

# Print the average rectified EMG for each contraction
for i, aremg in enumerate(aremg_values, 1):
    print(f"Contraction {i} Average Rectified EMG (AREMG): {aremg:.4f} mV")
# Plot average EMG & Force in column plot - HAHA
avg_emg = df['EMG'].mean()
avg_force = df['Force'].mean()

plt.figure(figsize=(6,6))
plt.bar(['Average EMG', 'Average Force'], [avg_emg, avg_force], color=['blue', 'green'])
plt.title('Average EMG vs Force')
plt.ylabel('Value')
plt.show()
# Trapezoidal integral = numpy trapz() | Cumulative inegral = numpy cumsum()
def plot_cumulative_integral(contractions, signal_column, sampling_rate):
    plt.figure(figsize=(10, 6))
    for contraction in contractions:
        signal=df[signal_column].values[contraction['start']:contraction['end']]
        cumulative_integral = np.cumsum(signal)/sampling_rate
        time = np.arange(0, len(signal)) / sampling_rate
        # plot each contraction
        plt.plot(time, cumulative_integral, label=f'Cumulative Integral of {contraction["title"]}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cumulative Integral (mV/s)')
    plt.title('Cumulative Integral of EMG Signal')
    plt.legend()
    plt.show()
plot_cumulative_integral(contractions, 'EMG_rms', sampling_rate)
# Fatiguing contractions