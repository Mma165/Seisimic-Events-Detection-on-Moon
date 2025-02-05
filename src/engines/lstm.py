import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from datetime import datetime, timedelta
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

class LSTMInferenceEngineV2:
    def __init__(
        self,
        model: tf.keras.Model,
        window_size: int = 1000,
        overlap: float = 0.8,  # Higher overlap for better detection
        detection_threshold: float = 0.5,
        scaler: StandardScaler = None
    ):
        self.model = model
        self.window_size = window_size
        self.overlap = overlap
        self.stride = int(window_size * (1 - overlap))
        self.detection_threshold = detection_threshold
        self.scaler = scaler

    def prepare_inference_windows(
        self,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, List[datetime]]:
        """
        Prepare sliding windows for inference and keep track of their timestamps.
        """
        windows = []
        window_times = []

        for i in range(0, len(data) - self.window_size, self.stride):
            window = data.iloc[i:i + self.window_size]

            if len(window) == self.window_size:
                # Store the center timestamp of the window
                center_time = window['time_abs'].iloc[self.window_size // 2]
                window_times.append(center_time)

                # Extract and normalize the amplitude data
                amplitude_window = window[['amplitude']].values
                windows.append(amplitude_window)

        X = np.array(windows)

        # Normalize each window independently
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped)
        X = X_scaled.reshape(X.shape)

        return X, window_times

    def detect_quakes(
        self,
        data: pd.DataFrame,
        plot_results: bool = True,
        figsize: Tuple[int, int] = (15, 10)
    ) -> pd.DataFrame:
        """
        Detect quakes in the input data and optionally plot the results.

        Args:
            data: DataFrame with columns ['time_abs', 'amplitude']
            plot_results: Whether to show the plot
            figsize: Size of the plot figure

        Returns:
            DataFrame with detected quake timestamps and confidence scores
        """
        # Prepare windows for inference
        X, window_times = self.prepare_inference_windows(data)

        # Get model predictions
        predictions = self.model.predict(X, verbose=0)

        # Find windows where prediction exceeds threshold
        detections = []
        for time, pred in zip(window_times, predictions):
            if pred[0] >= self.detection_threshold:
                detections.append({
                    'detection_time': time,
                    'confidence': float(pred[0])
                })

        # Convert detections to DataFrame
        detection_df = pd.DataFrame(detections)

        # Handle multiple valid windows
        if len(detection_df) > 0:
            detection_df = self._extend_detection_windows(detection_df)

        # Plot if requested
        if plot_results:
            self.plot_detections(data, detection_df, figsize)

        return detection_df

    def _extend_detection_windows(
        self,
        detections: pd.DataFrame,
        merge_window: timedelta = timedelta(seconds=100)
    ) -> pd.DataFrame:
        """
        Extend detection windows to highlight continuous activity
        until no detection is found for the next 60 seconds.
        """
        detections = detections.sort_values('detection_time')
        extended_windows = []
        current_window = [detections.iloc[0]['detection_time'], detections.iloc[0]['detection_time']]

        for i in range(1, len(detections)):
            if detections.iloc[i]['detection_time'] - current_window[1] <= merge_window:
                # Extend current window
                current_window[1] = detections.iloc[i]['detection_time']
            else:
                # Save the current window if it spans more than one detection
                if current_window[1] - current_window[0] >= timedelta(seconds=60):
                    extended_windows.append(current_window)
                # Start a new window
                current_window = [detections.iloc[i]['detection_time'], detections.iloc[i]['detection_time']]

        # Append the last window if valid
        if current_window[1] - current_window[0] >= timedelta(seconds=60):
            extended_windows.append(current_window)

        return pd.DataFrame(extended_windows, columns=['start_time', 'end_time'])

    def plot_detections(
        self,
        data: pd.DataFrame,
        detections: pd.DataFrame,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Plot the seismic data and highlight regions with detections.
        """
        plt.figure(figsize=figsize)

        # Plot amplitude data
        plt.plot(data['time_abs'], data['amplitude'], 'b-', alpha=0.6, label='Seismic Signal')

        # Highlight detected regions
        if len(detections) > 0:
            for _, detection in detections.iterrows():
                plt.axvspan(
                    detection['start_time'] - timedelta(seconds=60*10),
                    detection['end_time'] + timedelta(seconds=60*40),
                    color='yellow',
                    alpha=0.3,
                )

        plt.title('Seismic Signal with Detected Regions')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./images/detections.png')

def run_LSTM_engine(model_path: str, scaler_path, data_path: pd.DataFrame):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Initialize inference engine
    engine = LSTMInferenceEngineV2(
        model=model,
        window_size=1000,
        overlap=0.8,
        detection_threshold=0.5,
        scaler=scaler  # Use the same scaler used during training
    )

    def load_dataset(data_df):
        data_df.rename(columns={'time_abs(%Y-%m-%dT%H:%M:%S.%f)': 'time_abs'}, inplace=True)
        data_df.drop(columns=['time_rel(sec)'], inplace=True)
        data_df['time_abs'] = pd.to_datetime(data_df['time_abs'], format='%Y-%m-%dT%H:%M:%S.%f')
        data_df = data_df.rename(columns={'velocity(m/s)': 'amplitude'})
        return data_df

    data = load_dataset(pd.read_csv(data_path))

    # Run detection
    detections = engine.detect_quakes(
        data=data,
        plot_results=True,
        figsize=(15, 10)
    )

    print(detections)
