"""
Script: Suricata And Wireshark 
Description:  Creates new coloumn in wireshark to indicate wheter an alert was triggerd by suricata
              Column gets created and named label with 1s indicateing were the suricata rule was generated from.
Author: Matt Smith,
Rev: 07
Created: 07/03/24

"""

import pandas as pd

def process_alerts(suricata_alerts, wireshark_data, output_file):
    # Read Suricata alert data
    suricata_df = pd.read_csv(suricata_alerts)

    # Read Wireshark data
    wireshark_df = pd.read_csv(wireshark_data)

    # Create a new column in Wireshark data to indicate whether an alert is triggered
    wireshark_df['label'] = 0

    # Iterate through Suricata alerts and find matching rows in Wireshark data
    for index, alert_row in suricata_df.iterrows():
        matching_rows = wireshark_df[
            (wireshark_df['Source'] == alert_row['src_ip']) &
            (wireshark_df['Destination'] == alert_row['dest_ip']) &
            (wireshark_df['Protocol'] == alert_row['proto']) &
            (wireshark_df['Time'] == alert_row['timestamp'])
        ].index

        # Check if any matching rows are found and the timestamps match exactly
        if len(matching_rows) > 0 and wireshark_df.loc[matching_rows[0], 'Time'] == alert_row['timestamp']:
            # Mark matching rows with 1 in the 'label' column
            wireshark_df.loc[matching_rows, 'label'] = 1
        else:
            # Break if timestamps don't match exactly
            print("Timestamps don't match exactly. Breaking.")
            break

    # Save the result to a new CSV file
    wireshark_df.to_csv(output_file, index=False)
    print(f"Processed data written to {output_file}")

if __name__ == "__main__":
    # Example usage
    suricata_alerts_file = "model_data/Other-data/ICMP-suricata-test.csv"
    wireshark_data_file = "test.csv"
    output_file = "model_data/Testing_data/ICMP-Test.csv"
    process_alerts(suricata_alerts_file, wireshark_data_file, output_file)

