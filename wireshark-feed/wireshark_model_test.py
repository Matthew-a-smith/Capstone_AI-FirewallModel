import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import time
import csv
import subprocess
import threading
import shutil
from pandas.errors import EmptyDataError

def capture_tcp_traffic(csv_path):
    tshark_cmd = [
        "tshark",
        "-i", "wlan0",  # Specify the interface you want to capture traffic on
        "-Y", "tcp",  # Filter for TCP traffic
        "-E", "header=y",  # Add header to output
        "-T", "fields",  # Output fields
        "-E", "separator=, ",  # CSV separator
        "-e", "ip.dst",
        "-e", "ip.src",
        "-e", "tcp.dstport",
        "-e", "tcp.srcport",
        "-e", "tcp.window_size_value",
        "-e", "tcp.flags",
    ]

    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["ip.dst", "ip.src", "tcp.dstport", "tcp.srcport", "tcp.window_size_value", "tcp.flags"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        try:
            while True:
                subprocess.run(tshark_cmd, check=True, stdout=csvfile, universal_newlines=True)
                csvfile.flush()
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nTCP traffic capture stopped.")

def make_predictions(df, models, threshold=0.95):
    input_tensors = {name: tf.convert_to_tensor(df[name].values) for name in df.columns}
    predictions = [model.predict(input_tensors) for model in models]
    return predictions

def process_and_predict(csv_path, output_file_path, models, threshold=0.95, batch_size=100):
    dtypes = {
        "ip.dst": str,
        "ip.src": str,
        "tcp.dstport": int,
        "tcp.srcport": int,
        "tcp.window_size_value": int,
        "tcp.flags": str
    }

    while True:
        try:
            df = pd.read_csv(csv_path, dtype=dtypes)
        except EmptyDataError:
            print("No data available in CSV file. Waiting for data...")
            time.sleep(10)  # Wait for 10 seconds before checking again
            continue

        complete_rows = df.dropna()  # Remove rows with missing values

        if len(complete_rows) >= batch_size:
            predictions = make_predictions(complete_rows, models)
            with open(output_file_path, 'a') as f:
                f.write("Predictions:\n")
                for i, prediction in enumerate(predictions):
                    for idx, prob in enumerate(prediction[0]):
                        if prob >= threshold:
                            f.write(f"Model {i+1} predicts with {prob*100:.2f}% confidence:\n")
                            f.write(str(complete_rows.iloc[idx]) + "\n")

            # Garbage cleanup
            if os.path.getsize(csv_path) > 1000000:  # Check if CSV file size exceeds 1MB
                df.to_csv(output_file_path, mode='a', index=False, header=False)  # Append to the existing file
                print("Processed data appended to the existing file.")

        else:
            print("Waiting for more complete data to accumulate...")
            time.sleep(10)  # Wait for 10 seconds before checking again

if __name__ == "__main__":
    csv_path = "tcp_traffic.csv"
    output_file_path = "processed_data.txt"  # Changed to a different file for saving processed data
    threshold = 0.95
    batch_size = 100

    print("Data written to CSV file")

    model_paths = ["NMAP-inference_model.keras"]
    models = [keras.models.load_model(model_path) for model_path in model_paths]

    capture_thread = threading.Thread(target=capture_tcp_traffic, args=(csv_path,))
    capture_thread.start()

    process_thread = threading.Thread(target=process_and_predict, args=(csv_path, output_file_path, models, threshold, batch_size))
    process_thread.start()

    capture_thread.join()  # Wait for the capture thread to finish (this won't happen as it's an infinite loop)
    process_thread.join()  # Wait for the processing thread to finish (this won't happen as it's an infinite loop)
