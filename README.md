# Seismic Data Filtering And Detection

## Overview

This project tackles the challenge of filtering seismic data from Mars and the Moon to detect and send only the relevant impact data. Given the large and noisy nature of the data, we employed a **Mixture of Experts** approach, combining three distinct models and algorithms to accurately identify and process seismic events:

1. **[LSTM Model](./Development%20Notebooks/LSTM_Model.ipynb)** - Used for time-series analysis to detect potential seismic events based on historical patterns in the data.
2. **[YOLO Model](./Development%20Notebooks/YOLOv8_Model.ipynb)** - Applied for object detection and localization within seismic event clusters, further refining the relevant data.
3. **[Bandpass Algorithm](./Development%20Notebooks/Bandpass_Algorithm.ipynb)** - Utilized to filter noise and focus on seismic signals within a specific frequency range.

This hybrid solution ensures both accurate detection and efficient filtering of irrelevant or noisy data, enabling the transmission of only meaningful seismic events.

## Project Structure

The repository is structured as follows:

- **Development Notebooks/**: Contains all Jupyter notebook files used during the development and experimentation phases.
- **Models/**: Stores the pre-trained model weights for the LSTM, YOLO.
- **src/**: Contains the inference code that you can use to apply the models to new data.

## Inference Workflow

To run inference on a new seismic data file, follow these steps:

1. Clone the repository.
2. Navigate to the `src/` directory:
   ```bash
   cd src
   ```
2. Ensure that the necessary dependencies are installed by running:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your input seismic data CSV file in the appropriate location.
4. Print the help message to understand the input arguments:
   ```bash
   python inference.py --help
   ```
5. Run the inference script with the appropriate arguments:
   ```bash
   python inference.py lstm ./path/to/input/data.csv --model_path ../Models/LSTM_model.keras --scaler_path ../Models/LSTM_model_scaler.pkl 
   ```
   
The script will automatically process the input data using the LSTM model, YOLO model, or Bandpass filter to detect and highlight relevant seismic events. It will output both the relevant timestamps and plots of seismic event clusters for further analysis under `./src/images` directory.

## Key Features

- **Mixture of Experts** approach combining different models for higher accuracy.
- **LSTM** for time-series pattern recognition.
- **YOLO** for event localization within seismic clusters.
- **Bandpass Filter** for signal refinement and noise reduction.
- End-to-end pipeline for filtering, detecting, and clustering seismic events.

## Dependencies

The project relies on the following libraries and frameworks:

- TensorFlow (for LSTM model)
- PyTorch (for YOLO model)
- NumPy, Pandas (for data manipulation)
- SciPy (for Bandpass filter implementation)
- Matplotlib (for plotting detection clusters)

## How It Works

1. **LSTM Model**: Scans through the time-series data to detect potential seismic events.
2. **YOLO Model**: Identifies and localizes event clusters within the data.
3. **Bandpass Filter**: Cleans the data by removing irrelevant frequencies and focusing on seismic signals.
5. **Output**: Produces a plot with highlighted regions of interest and outputs the start and end timestamps of detected seismic events.
