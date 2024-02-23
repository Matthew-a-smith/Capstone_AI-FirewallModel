import pandas as pd

# Read labeled and unlabeled data
labeled_data = pd.read_csv("wireshark-data/nmap-ss.csv")
unlabeled_data = pd.read_csv("tcp_traffic.csv")

# Assign label of 1 to the labeled data
labeled_data['label'] = 1

# Add a new column called "label" to the unlabeled data and assign zeros
unlabeled_data['label'] = 0

# Append labeled data to unlabeled data
merged_data = unlabeled_data.append(labeled_data, ignore_index=True)

# Save the merged data
merged_data.to_csv("nmap-sn-training-data.csv", index=False)