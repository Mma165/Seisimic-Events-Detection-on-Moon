import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.signal import butter, filtfilt, correlate
import pickle
import matplotlib.pyplot as plt

# Function to load and preprocess data
def load_data(csv_path):
    """
    Load data from a CSV file.

    Parameters:
    - csv_path: Path to the CSV file.

    Returns:
    - DataFrame containing the data.
    """
    data = pd.read_csv(csv_path)
    return data

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to the data.

    Parameters:
    - data: Input signal data (1D array).
    - lowcut: Lower frequency bound for the filter.
    - highcut: Upper frequency bound for the filter.
    - fs: Sampling frequency of the data.
    - order: Order of the filter.

    Returns:
    - Filtered data (1D array).
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Function to preprocess data for LSTM
def preprocess_for_lstm(data, scaler):
    """
    Preprocess input data for the LSTM model using the provided scaler.

    Parameters:
    - data: Input data to be scaled (2D array or DataFrame).
    - scaler: Preloaded scaler object for preprocessing.

    Returns:
    - Scaled data ready for LSTM model prediction.
    """
    return scaler.transform(data)

# Function for cross-correlation plotting
def plot_detected_events(signal, template, cross_corr, threshold, sampling_rate):
    """
    Plot the detected seismic events using cross-correlation.
    """
    lags = np.arange(-len(template) + 1, len(signal))
    time = lags / sampling_rate

    # Plot cross-correlation
    plt.figure(figsize=(12, 6))
    plt.plot(time, cross_corr, label="Cross-correlation")

    # Highlight regions above the threshold
    for idx, value in enumerate(cross_corr):
        if value > threshold:
            plt.axvline(x=time[idx], color="red", alpha=0.5, label="Detected Event" if idx == 0 else None)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Cross-correlation")
    plt.title("Seismic Event Detection using Ensemble Learning")
    plt.legend()
    plt.grid()
    plt.show()

# Main ensemble function
def ensemble_predict(csv_path, lstm_model_path, scaler_path, lowcut, highcut, fs, order=5, threshold=0.5):
    """
    Perform ensemble prediction using LSTM and Bandpass Filter.

    Parameters:
    - csv_path: Path to the input CSV file.
    - lstm_model_path: Path to the LSTM model file.
    - scaler_path: Path to the scaler file for LSTM preprocessing.
    - lowcut: Lower frequency for Bandpass Filter.
    - highcut: Upper frequency for Bandpass Filter.
    - fs: Sampling frequency of the data.
    - order: Order of the Bandpass Filter.
    - threshold: Threshold for event detection.

    Returns:
    - Ensemble prediction.
    """
    # Load data
    data = load_data(csv_path)
    signal = data['velocity(m/s)'].values  # Replace with the correct signal column

    # Apply Bandpass Filter
    filtered_signal = bandpass_filter(signal, lowcut, highcut, fs, order)

    # Load LSTM model
    lstm_model = tf.keras.models.load_model(lstm_model_path)

    # Load scaler
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Preprocess for LSTM
    lstm_input = filtered_signal.reshape(-1, 1)  # Ensure 2D input
    lstm_input_scaled = preprocess_for_lstm(lstm_input, scaler)
    lstm_prediction = lstm_model.predict(lstm_input_scaled).mean()

    # Bandpass Prediction (e.g., energy-based)
    bandpass_prediction = np.mean(filtered_signal)

    # Ensemble Voting
    if bandpass_prediction > lstm_prediction:
        ensemble_prediction = bandpass_prediction
    else:
        ensemble_prediction = lstm_prediction

    # Cross-Correlation with template
    template = np.sin(2 * np.pi * 0.1 * np.arange(0, 100))  # Replace with your template
    cross_corr = correlate(filtered_signal, template, mode='full')
    cross_corr /= np.max(np.abs(cross_corr))  # Normalize

    # Plot detected events
    plot_detected_events(filtered_signal, template, cross_corr, threshold, fs)

    return ensemble_prediction

# Example usage
if _name_ == "_main_":
    # Parameters
    csv_path = "first.csv"
    lstm_model_path = r"C:/Users/omark/Downloads/seismic-quakes-detection-main/Models/LSTM_model.keras"
    scaler_path = r"C:/Users/omark/Downloads/seismic-quakes-detection-main/Models/LSTM_model_scaler.pkl"
    lowcut = 0.1
    highcut = 10.0
    fs = 100
    order = 5
    threshold = 0.5

    # Perform ensemble prediction
    prediction = ensemble_predict(csv_path, lstm_model_path, scaler_path, lowcut, highcut, fs, order, threshold)
    print("Ensemble Prediction:", prediction)