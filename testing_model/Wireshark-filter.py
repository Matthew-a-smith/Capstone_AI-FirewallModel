import pandas as pd

def clean_Time_and_save(df, output_file):
    # Drop columns 'No.' and 'Info'
    df = df.drop(columns=['No.'])
    
    # Save filtered data to a single file
    df.to_csv(output_file, index=False)
    
    print(f"Filtered data written to {output_file}")
    return df

def main(input_csv_file, output_file):
    # Load the data
    df = pd.read_csv(input_csv_file)

    # Assuming undesired_protocols is defined earlier in your script
    undesired_protocols = ['TLSv1.3', 'ARP', 'DHCPv6', 'SSDP', 'TPLINK-SMARTHOME/JSON', 'ICMPv6', 'MDNS', 'QUIC']

    # Drop rows where Protocol is in the list of undesired protocols
    filtered_df = df[~df['Protocol'].isin(undesired_protocols)]

    # Call the function to clean Time and save
    filtered_df = clean_Time_and_save(filtered_df, output_file)

if __name__ == "__main__":
    # Example usage
    formatted_csv_file = "home-network-feb06.csv"
    output_file = "model_data/Training_data/Wireshaark-test-feb9.csv"
    main(formatted_csv_file, output_file)
