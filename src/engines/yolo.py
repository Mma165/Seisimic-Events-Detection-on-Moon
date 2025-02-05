import pandas as pd
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO 
import os
import glob

output_dir = "./images/"
os.makedirs(output_dir, exist_ok=True)

def run_YOLO_engine(model_path, data_path):
    # Step 1: Load the CSV file
    df = pd.read_csv(data_path)

    # Step 2: Plot the velocity (m/s) column in black to simulate a seismograph
    plt.figure(figsize=(10, 5))
    plt.plot(df['velocity(m/s)'], color='black')  # Black color plot
    plt.title('Seismograph of Velocity (m/s)')
    plt.xlabel('Time')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)

    # Save the plot as an image file to use it with YOLOv8
    seismograph_image_path = os.path.join(output_dir, data_path.split('/')[-1].replace('.csv', '.png'))
    plt.savefig(seismograph_image_path, bbox_inches='tight')
    plt.close()

    # Load the YOLO model
    model = YOLO(model_path)

    # Run batched inference on a list of images
    results = model([seismograph_image_path], stream=True)

    # Process results generator
    for result in results:
        boxes = result.boxes
        result.save(filename=output_dir + 'detections.png')
