import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras

os.environ["KERAS_BACKEND"] = "tensorflow"

# Load the saved models
model_paths = [
    "AI_Model/Validated-Models/ICMP-inference_model.keras",
    "AI_Model/Validated-Models/NMAP-inference_model.keras"
]

models = [keras.models.load_model(model_path) for model_path in model_paths]

# Load the CSV file
csv_path = "home-network-feb09.csv"  
df = pd.read_csv(csv_path)

# Drop specific columns
df = df.drop(columns=['No.'])

# Filter DataFrame based on relevant protocol (e.g., TCP)
relevant_protocols = ['TCP', 'DNS', 'ICMP']  # Adjust this list as per your relevant protocols
df_filtered = df[df['Protocol'].isin(relevant_protocols)].copy()  # Create a copy to avoid the warning

# Convert DataFrame columns to TensorFlow tensors
input_tensors = {name: tf.convert_to_tensor(df_filtered[name].values) for name in df_filtered.columns}

# Make predictions using each model
predictions = [model.predict(input_tensors) * 100 for model in models]  # Convert to percentage

# Add the predictions to the filtered DataFrame using .loc
for i, prediction in enumerate(predictions):
    df_filtered.loc[:, f'probability_model{i+1}'] = prediction

# Sort the DataFrame by the 'probability' column in descending order for each model
df_sorted_models = [df_filtered.sort_values(by=f'probability_model{i+1}', ascending=False).head(10) for i in range(len(models))]

# Write the sorted data to the output file
output_file_path = "feb-09-test-predictions.txt"  # Replace with the desired path for the output file
with open(output_file_path, 'w') as f:
    for i, df_sorted_model in enumerate(df_sorted_models):
        f.write(f"Model {i+1} Top 10 Predictions:\n")
        for idx, row in df_sorted_model.iterrows():
            f.write(f"No: {idx}, Time:{row['Time']}, Source:{row['Source']}, Protocol:{row['Protocol']}, Info:{row['Info']}, (Model {i+1}):{row[f'probability_model{i+1}']:.2f}%\n")
