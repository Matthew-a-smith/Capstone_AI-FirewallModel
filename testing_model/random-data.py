import csv
import random

# Function to generate a random IP address within a predefined range
def generate_random_ip_in_range():
    # Common private IP address ranges
    private_ranges = [
        "10.0.0.0",        # Private network
        "172.16.0.0",      # Private network
        "192.168.0.0"      # Private network
    ]
    # Select a random private range
    ip_range = random.choice(private_ranges)

    # Split the IP address into octets
    octets = ip_range.split('.')
    # Generate random values for the last two octets
    third_octet = random.randint(1, 54)
    fourth_octet = random.randint(1, 254)
    # Combine to form the IP address
    random_ip = f"{octets[0]}.{octets[1]}.{third_octet}.{fourth_octet}"

    return random_ip

# Input and output file paths
input_file = "model_data/Testing_data/NMAP-Test.csv"
output_file = "test1.csv"

# Open input and output files
with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Write header
    header = next(reader)
    writer.writerow(header)
    
    # Process and write each row
    for row in reader:
        row[1] = generate_random_ip_in_range()  # Replace source IP
        row[2] = generate_random_ip_in_range()  # Replace destination IP
        writer.writerow(row)

print("Data successfully processed and written to", output_file)
