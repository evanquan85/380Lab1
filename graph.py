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
# Average EMG and Force values for each contraction
contractions2 = ['Contraction 25%', 'Contraction 50%', 'Contraction 75%', 'Contraction 100%']
average_emg = [
    df['EMG_rms'].iloc[45500:47750].mean(),
    df['EMG_rms'].iloc[54500:56750].mean(),
    df['EMG_rms'].iloc[69100:71350].mean(),
    df['EMG_rms'].iloc[78800:81050].mean()
]
average_force = [
    df['Force'].iloc[45500:47750].mean(),
    df['Force'].iloc[54500:56750].mean(),
    df['Force'].iloc[69100:71350].mean(),
    df['Force'].iloc[78800:81050].mean()
]
# Create a figure for column graph of avg EMG & Force per contraction
n = len(contractions2)
bar_width = 0.3
x = np.arange(n)
# Create the figure and axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the bars for EMG on the primary y-axis
bar1 = ax1.bar(x - bar_width/2, average_emg, bar_width, label='Average EMG (mV)', color='blue', alpha=0.7)
ax1.set_ylabel('Average EMG (mV)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a secondary y-axis for Force
ax2 = ax1.twinx()
bar2 = ax2.bar(x + bar_width/2, average_force, bar_width, label='Average Force (kg)', color='red', alpha=0.7)
ax2.set_ylabel('Average Force (kg)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Set x-axis labels and title
ax1.set_xlabel('Contractions')
ax1.set_title('Average EMG and Force for Each Contraction')
ax1.set_xticks(x)
ax1.set_xticklabels(contractions2)

emg_max_ax1=max(average_emg)
ax1.set_ylim(0, emg_max_ax1 * 1.2)
force_max_ax1 = max(average_force)
ax2.set_ylim(0, force_max_ax1 * 1.2)
# Combine legends from both axes
bars = [bar1, bar2]
labels = [bar.get_label() for bar in bars]
ax1.legend(bars, labels, loc='upper left')

# Show the plot
plt.tight_layout()
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
# Identify series of fatiguing contractions
# total samples = 405529
fig, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(df['EMG'])
ax3.set_xlim(100000, 120000)
plt.show()
start_index = 100000
filtered_df = df.iloc[start_index:]
def identify_contractions(signal, threshold, min_duration, sampling_rate, start_offset):
    """
    Identify contraction start and end points based on a signal threshold.
    """
    above_threshold = signal > threshold
    indices = np.where(np.diff(above_threshold.astype(int)) != 0)[0]
    contractions = []

    for i in range(0, len(indices), 2):
        if i + 1 < len(indices):
            start, end = indices[i], indices[i + 1]
            duration = (end - start) / sampling_rate
            if duration >= min_duration:
                contractions.append({'start': start + start_offset, 'end': end + start_offset})
    return contractions

# Parameters
threshold = 0.33  # Threshold for identifying contractions (adjust as needed)
min_duration = 0.5  # Minimum duration of a contraction in seconds
start_offset = start_index
contractions = identify_contractions(df['Force'], threshold, min_duration, sampling_rate, start_offset)


# Calculate AREMG for each contraction
aremg_values = []
for contraction in contractions:
    start, end = contraction['start'], contraction['end']
    aremg = df['EMG'].iloc[start:end].mean()
    aremg_values.append(aremg)

# Display results
for i, contraction in enumerate(contractions):
    start, end = contraction['start'], contraction['end']
    print(f"Contraction {i + 1}: Start = {start}, End = {end}, AREMG = {aremg_values[i]:.4f} mV")


