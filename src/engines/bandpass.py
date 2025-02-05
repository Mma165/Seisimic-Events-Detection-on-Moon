# Detection of Seismic Events using Bandpass Filter.
# 
# Bandpass filtering is used to clean the signal by removing noise, while matched filtering, using Signal-to-Noise Ratio (SNR) and template matching, detects events that resemble predefined seismic signatures. This improves the clarity and precision of event detection, particularly for faint signals.

## Install and import dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, correlate
import os

## Bandpass Implementation

# Filters out frequencies outside the desired range to remove noise and retain relevant signals.
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')  # Create filter coefficients
    filtered_data = filtfilt(b, a, data.fillna(0))  # Apply filter and handle missing values
    return filtered_data

# This template is used for cross-correlation with real data.
def create_synthetic_template(sampling_rate=6.625, frequency=6.6, duration=20):
    t = np.linspace(0, duration, int(sampling_rate * duration))  # Time vector
    template = np.sin(2 * np.pi * frequency * t)  # Create sine wave
    return template

# Finds similarity between the seismic data and the template to detect events.
def apply_matched_filter(data, template):
    cross_corr = correlate(data, template, mode='same')  # Compute cross-correlation
    threshold = 0.5 * np.max(cross_corr)  # Set threshold for event detection
    return cross_corr, threshold

# Groups nearby events and extends their window to capture full event length.
def extend_event_detection(cross_correlation, threshold, extension=500, proximity=1000):
    detected_event_indices = np.where(cross_correlation > threshold)[0]
    extended_event_indices = []

    if len(detected_event_indices) > 0:
        start = detected_event_indices[0]
        end = start

        for i in range(1, len(detected_event_indices)):
            if detected_event_indices[i] - end <= extension:
                end = detected_event_indices[i]
            else:
                if extended_event_indices and start - extended_event_indices[-1][1] <= proximity:
                    extended_event_indices[-1] = (extended_event_indices[-1][0], end)
                else:
                    extended_event_indices.append((start, end))
                start = detected_event_indices[i]
                end = start

        # Save the final event
        extended_event_indices.append((start, end))

    return extended_event_indices

# Function to plot detected events and filter out empty plots.
# If there are no detected events, no plot will be generated.
def plot_events(data, time_column, signal_column, events, file_name):
    if events:  # Only plot if there are detected events
        for idx, (start, end) in enumerate(events):
            plt.figure(figsize=(10, 4))
            plt.plot(data[time_column][start:end], data[signal_column][start:end], label="Event Signal")
            plt.title(f"Detected Event {idx+1} - {file_name}")
            plt.xlabel('Time (seconds)')
            plt.ylabel('Signal')
            plt.legend()
            plt.savefig(f"./images/{file_name}_event_{idx+1}.png")

# Function to process the seismic data files.
# Each file is filtered, cross-correlated, and events are detected and plotted.
def run_bandpass_engine(data_path, sampling_rate=6.625, lowcut=0.4, highcut=2.0):
    file_name = os.path.basename(data_path)
    data = pd.read_csv(data_path)  # Load data

    template = create_synthetic_template(sampling_rate, 6.6, 20)  # Create synthetic template
    filtered_data = bandpass_filter(data['velocity(m/s)'], lowcut, highcut, sampling_rate)  # Filter data
    cross_corr, threshold = apply_matched_filter(filtered_data, template)  # Apply matched filter
    
    # Detect and extend event windows
    extended_event_windows = extend_event_detection(cross_corr, threshold)
    
    if extended_event_windows:  # Only plot if events are detected
        # Plot full signal with event regions highlighted
        plt.figure(figsize=(15, 6))
        plt.plot(data['time_rel(sec)'], cross_corr, label="Cross-correlation")
        for start, end in extended_event_windows:
            plt.axvspan(data['time_rel(sec)'].iloc[start], data['time_rel(sec)'].iloc[end], color='red', alpha=0.5)
        
        plt.title(f"Matched Filter Detection - {file_name}")
        plt.xlabel('Time (seconds)')
        plt.ylabel('Cross-correlation')
        plt.legend()
        plt.savefig(f"./images/{file_name}_detections.png")
        
        # Plot each cropped event
        plot_events(data, 'time_rel(sec)', 'velocity(m/s)', extended_event_windows, file_name)
    else:
        print(f"No events detected in {file_name}, skipping plot.")
