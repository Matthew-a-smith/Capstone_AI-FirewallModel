import csv
import random
import tempfile
import shutil

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

# Function to generate a random port number
def generate_random_port():
    return random.randint(80, 5555)  # Choosing ports from the dynamic and/or private port range

# Input file path
input_file = "nmap-sn-training-data.csv"

# Create a temporary file to store the modified data
temp_output = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='', encoding='utf-8')

# Open input file
with open(input_file, mode='r', newline='', encoding='utf-8') as infile, temp_output:
    reader = csv.reader(infile)
    writer = csv.writer(temp_output)
    
    # Write header
    header = next(reader)
    writer.writerow(header)
    
    # Process and write each row
    for row in reader:
        #row[0] = generate_random_ip_in_range()  # Replace source IP
        #row[1] = generate_random_ip_in_range()  # Replace destination IP
        row[2] = generate_random_port()  # Replace destination port
        row[3] = generate_random_port()  # Replace source port
        writer.writerow(row)

# Replace the original file with the modified temporary file
shutil.move(temp_output.name, input_file)

print("Data successfully processed and modified in the input file.")
