import torch
from scapy.all import *

def process_pcap(file_path, max_length=22):  # Set the desired maximum length
    packets = rdpcap(file_path)
    dns_data = [packet[DNS].qd.qname.decode('utf-8') for packet in packets if DNS in packet and packet.haslayer(DNSQR)]

    # Pad sequences to a fixed length
    dns_data_padded = [query_name[:max_length].ljust(max_length, '\0') for query_name in dns_data]

    # Convert DNS data to PyTorch tensor
    tensor_data = torch.tensor([list(map(ord, query_name)) for query_name in dns_data_padded], dtype=torch.long)

    return tensor_data

if __name__ == '__main__':
    tensor_data = process_pcap('data.pcapng')
    torch.save(tensor_data, 'dataset.pt')
    
    print("Data processing completed. Tensor data saved as 'dataset.pt'")
