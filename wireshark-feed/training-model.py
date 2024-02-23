import tensorflow as tf

# Load the inference model
ICMP_inference_model = "NMAP-inference_model.keras"
loaded_model = tf.keras.models.load_model(ICMP_inference_model)

# Define function for making predictions
def make_prediction(sample):
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
    predictions = loaded_model.predict(input_dict)
    return predictions[0][0]

# Define sample data
samples = [
    {
        "ip.dst": "192.168.2.15",
        "ip.src": "192.168.2.23",
        "tcp.dstport": 8009,
        "tcp.srcport": 42394,
        "tcp.window_size_value": 1024,
        "tcp.flags": "0x0000",
    },
    {
        "ip.dst": "192.168.2.15",
        "ip.src": "192.168.2.23",
        "tcp.dstport": 8009,
        "tcp.srcport": 42394,
        "tcp.window_size_value": 501,
        "tcp.flags": "0x0014",
    },
    {
        "ip.dst": "192.168.2.15",
        "ip.src": "192.168.2.23",
        "tcp.dstport": 8009,
        "tcp.srcport": 42394,
        "tcp.window_size_value": 64000,
        "tcp.flags": "0x0014",
    }
]

# Make predictions for each sample
for i, sample in enumerate(samples, start=1):
    prediction = make_prediction(sample)
    print(f"Prediction for Sample {i}: This event has a {100 * prediction:.2f}% probability of being of a certain type.")
