"""
Script: Suricata Alert extract
Description:  Extracts alert data from a suricata log file and outputs it in CSV format.
Author: Matt Smith,
Rev: 04
Created: 06/03/24

"""

import json
import pandas as pd

def parse_suricata_log(json_log):
    try:
        return json.loads(json_log)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

def extract_all_fields(log_entry):
    return log_entry

def parse_and_filter_data(json_file, output_file):
    with open(json_file, 'r') as f:
        json_logs = f.readlines()

    logs = [parse_suricata_log(log) for log in json_logs]
    df = pd.DataFrame(logs)

    # Include rows with 'alert' event_type and rows with NaN event_type
    df = df[(df['event_type'] == 'alert') | df['event_type'].isna()]

    # Extract relevant fields
    relevant_fields = ['timestamp', 'event_type', 'src_ip', 'dest_ip', 'proto', 'src_port', 'dest_port', 'alert']

    # Clean up timestamp feature
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['timestamp'] = df['timestamp'].dt.strftime('%Y/%j %H:%M:%S.%f')

    # Save filtered data to a single file
    df[relevant_fields].to_csv(output_file, index=False)
    print(f"Filtered data written to {output_file}")

if __name__ == "__main__":
    # Example usage
    suricata_json_file = "suricacta_logs/eve.json"
    output_csv_file = "model_data/Other-data/NMAP-suricata-test.csv"
    parse_and_filter_data(suricata_json_file, output_csv_file)

