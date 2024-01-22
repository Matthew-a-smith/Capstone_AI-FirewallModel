process_data.py -- 
serves as the initial step in the firewall creation pipeline. It takes a Wireshark data file (in .pcapng format) as input and extracts DNS features for identifying patterns and potential threats. 
The script employs a simplified processing logic to convert this data into a PyTorch tensor format. Specifically, it maps the characters in the DNS query names to their ASCII values and constructs a tensor. 
The resulting tensor is then saved for later use in training the firewall model.

NOTES: Save wireshark fire as data.pcapng save it in the same direcotry, this is to avoid file errors 

train_model.py --

script is responsible for training the neural network-based firewall model. It loads the preprocessed tensor data, initializes the model architecture with an encoder and decoder, and utilizes Mean Squared Error loss for training. 
The optimization process is handled by the Adam optimizer. The trained model is saved as firewall_model.pth for future use in predicting and creating firewall rules.
