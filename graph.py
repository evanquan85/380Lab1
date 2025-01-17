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
df['EMG'] = df['EMG'].rolling(window=window_size, center=True).mean()
def calculate_rms(signal):
    return np.sqrt(np.mean(signal**2))
df['EMG'] = df['EMG'].rolling(window=1000).apply(calculate_rms, raw=True)
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
    df['EMG'].iloc[45500:47750].mean(),
    df['EMG'].iloc[54500:56750].mean(),
    df['EMG'].iloc[69100:71350].mean(),
    df['EMG'].iloc[78800:81050].mean()
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
plot_cumulative_integral(contractions, 'EMG', sampling_rate)
# Fatiguing contractions
# Identify series of fatiguing contractions
# total samples = 405529 - use df.shape
def identify_contractions(signal, threshold, min_duration, sampling_rate, start_offset, end_index):
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
                start_adj = start + start_offset
                end_adj = min(end + start_offset, end_index)  # Crop end to max_index
                if start_adj < end_index:  # Only include valid contractions
                    contractions.append({'start': start_adj, 'end': end_adj})
    return contractions


# Parameters
threshold = 0.4  # Threshold for identifying contractions (adjust as needed)
min_duration = 0.5  # Minimum duration of a contraction in seconds
start_index = 110000
end_index = 405529
start_offset = start_index
contractions = identify_contractions(df['Force'], threshold, min_duration, sampling_rate, start_offset, end_index)


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

# Visualize average EMG of fatiguing contractions over time (column)
contraction_numbers = list(range(1, len(aremg_values) + 1))  # Contraction numbers (1, 2, 3, ...)
average_emg_values = [value for value in aremg_values if not np.isnan(value)]  # Filter out NaN values

plt.figure(figsize=(12, 8))
plt.bar(contraction_numbers, average_emg_values, color='blue', alpha=0.7)
plt.xlabel('Contraction Number')
plt.ylabel('Average EMG Value (mV)')
plt.title('Average EMG Value vs Contraction Number')
plt.xticks(contraction_numbers)  # Ensure x-axis matches contraction numbers
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()

# Identify first, 6th, last contractions from fatigue trial
# Calculate cumulative integral for last 2 seconds of each contraction
# 2 seconds = 4000Hz
"""
Contraction 1: Start = 129659, End = 137884, AREMG = 0.1261 mV - 2 seconds = [133884:137884]
Contraction 6: Start = 188303, End = 192398, AREMG = 0.0465 mV - 2 seconds = [188398:192398]
Contraction 25: Start = 401431, End = 405529, AREMG = 0.0621 mV - 2 seconds = [401529:405529]
"""
time_interval = 1/sampling_rate

last_2_second_ranges = {
    "Contraction 1": (133884, 137884),
    "Contraction 6": (188398, 192398),
    "Contraction 25": (401529, 405529),
}

# Calculate cumulative integral for each contraction
cumulative_integrals = {}
for name, (start, end) in last_2_second_ranges.items():
    segment = df['EMG'].iloc[start:end]
    cumulative_integral = np.cumsum(segment) * time_interval  # Convert to mV * s
    cumulative_integrals[name] = cumulative_integral

plt.figure(figsize=(10, 6))

for name, (start, end) in last_2_second_ranges.items():
    segment = df['EMG'].iloc[start:end]
    cumulative_integral = np.cumsum(segment) * time_interval  # Convert to mV * s
    time = np.arange(0, len(segment)) * time_interval  # Time in seconds for the segment

    # Smooth scatter plot
    plt.plot(time, cumulative_integral, label=f'{name} Cumulative Integral')

# Customize the plot
plt.xlabel('Time (seconds)')
plt.ylabel('Cumulative Integral (mV*s)')
plt.title('Cumulative Integral of Last 2 Seconds for Selected Contractions')
plt.legend()
plt.grid(alpha=0.5)
plt.tight_layout()

# Show the plot
plt.show()