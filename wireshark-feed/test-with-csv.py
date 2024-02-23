import json
import pandas as pd
import tensorflow as tf

# Load the inference model
ICMP_inference_model = "NMAP-inference_model.keras"
loaded_model = tf.keras.models.load_model(ICMP_inference_model)

# Function to make prediction on the entire dataset
def make_predictions(data):
    input_data = {name: tf.convert_to_tensor(data[name].values.reshape(-1, 1)) for name in data.columns}
    predictions = loaded_model.predict(input_data)
    return predictions.flatten()

# Load data from CSV file
csv_file = "tcp_traffic.csv"  # Update with your CSV file path
data = pd.read_csv(csv_file)

# Make predictions for the entire dataset
all_predictions = make_predictions(data)

# Filter predictions with probability greater than 95%
high_probability_predictions = [(index, prediction) for index, prediction in enumerate(all_predictions) if prediction > 0.95]

# If no predictions are over 95%, return the first 20 with the highest results
if not high_probability_predictions:
    high_probability_predictions = sorted(enumerate(all_predictions), key=lambda x: x[1], reverse=True)[:20]

# Prepare output data
output_data = [{"Index": index, "Prediction": prediction * 100, "Details": data.iloc[index].to_dict()} for index, prediction in high_probability_predictions]

# Write predictions with additional information to a JSON file
output_file = "predictions.json"  # Update with your output file path
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)

print("Predictions have been written to", output_file)

