# A CLI interface to run the inference engines on seismic data files.
# The CLI supports three modes: bandpass, YOLO, and LSTM.
# The user can specify the path to the data file, the engine to use, and engine specific parameters.
# The CLI will load the data, run the specified engine, and save the results to the output directory.

from engines.bandpass import run_bandpass_engine
from engines.lstm import run_LSTM_engine
from engines.yolo import run_YOLO_engine
import argparse
import os

os.makedirs('./images', exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Run inference on seismic data files.')
    parser.add_argument('engine', type=str, choices=['bandpass', 'lstm', 'yolo'], help='Inference engine to use.')
    parser.add_argument('data_path', type=str, help='Path to the seismic data file.')
    parser.add_argument('--model_path', type=str, help='Path to the model file.')
    parser.add_argument('--scaler_path', type=str, help='Path to the scaler file.')
    args = parser.parse_args()

    if args.engine == 'bandpass':
        run_bandpass_engine(args.data_path)
    elif args.engine == 'lstm':
        if not args.model_path or not args.scaler_path:
            print("Model path and scaler path are required for LSTM engine.")
            return
        run_LSTM_engine(args.model_path, args.scaler_path, args.data_path)
    elif args.engine == 'yolo':
        if not args.model_path:
            print("Model path is required for YOLO engine.")
            return
        run_YOLO_engine(args.model_path, args.data_path)

if __name__ == '__main__':
    main()
