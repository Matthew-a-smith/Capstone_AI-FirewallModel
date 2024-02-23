import csv
import subprocess
import time

def capture_tcp_traffic():
    # Define the tshark command to capture TCP traffic and output as CSV
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

    # Open CSV file for writing
    with open("tcp_traffic.csv", "w", newline="") as csvfile:
        fieldnames = []  # Define CSV header field names
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write CSV header
        writer.writeheader()

        # Continuously capture TCP traffic and write to CSV
        try:
            while True:
                subprocess.run(tshark_cmd, check=True, stdout=csvfile, universal_newlines=True)
                csvfile.flush()  # Flush the buffer to ensure data is written to the file
                time.sleep(5)  # Sleep for 5 seconds before capturing again
        except KeyboardInterrupt:
            print("\nTCP traffic capture stopped.")

if __name__ == "__main__":
    capture_tcp_traffic()
