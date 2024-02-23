# Import necessary libraries
import os
import tensorflow as tf
import pandas as pd
import keras
from keras import regularizers
from keras.utils import FeatureSpace
from keras.callbacks import Callback
from keras.layers import Layer
# Set Keras backend to TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"



# Load the dataset
file_url = "nmap-sn-training-data.csv"
dataframe = pd.read_csv(file_url)

# Display the shape and first few rows of the dataset
print(dataframe.shape)
dataframe.head()

# Split the data into training and validation sets
val_dataframe = dataframe.sample(frac=0.2, random_state=1337)  # Remove this line
train_dataframe = dataframe.drop(val_dataframe.index)  # Remove this line

# Display the number of samples for training and validation
print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

# Convert dataframes to TensorFlow datasets
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("label")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

# Display a sample input and target from the training dataset
for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

# Batch the datasets
train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)
#ip.dst,ip.src,tcp.dstport,tcp.srcport,tcp.window_size_value,tcp.flags,tcp.options.mss_val
feature_space = FeatureSpace(
    features={
        "ip.dst": FeatureSpace.string_categorical(num_oov_indices=1),
        "ip.src": FeatureSpace.string_categorical(num_oov_indices=1),
        "tcp.dstport": FeatureSpace.integer_categorical(num_oov_indices=1),
        "tcp.srcport": FeatureSpace.integer_categorical(num_oov_indices=1),
        "tcp.window_size_value": FeatureSpace.integer_categorical(num_oov_indices=1),
        "tcp.flags": FeatureSpace.string_categorical(num_oov_indices=1),        
    },
    crosses=[
        FeatureSpace.cross(feature_names=("ip.dst", "ip.src"), crossing_dim=64),
        FeatureSpace.cross(feature_names=("ip.src", "tcp.flags"), crossing_dim=64),
    ],
    output_mode="concat",
)

# Adapt the custom layer to your data
train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels )

for x, _ in train_ds.take(1):
    preprocessed_x = feature_space(x)
    print("preprocessed_x.shape:", preprocessed_x.shape)
    print("preprocessed_x.dtype:", preprocessed_x.dtype)

preprocessed_train_ds = train_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

preprocessed_val_ds = val_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)

dict_inputs = feature_space.get_inputs()
encoded_features = feature_space.get_encoded_features()

x = keras.layers.Dense(64, activation="relu")(encoded_features)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(32, activation="relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(1, activation="sigmoid")(x)

training_model = keras.Model(inputs=encoded_features, outputs=predictions)
training_model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

inference_model = keras.Model(inputs=dict_inputs, outputs=predictions)

# Train the model with ProbabilityThresholdStop callback
history = training_model.fit(
    preprocessed_train_ds,
    epochs=30,
    validation_data=preprocessed_val_ds,
    verbose=2,
)

# Save the trained and inference models
ICMP_training_model = "NMAP-trained_model.keras"
training_model.save(ICMP_training_model)

ICMP_inference_model = "NMAP-inference_model.keras"
inference_model.save(ICMP_inference_model)

# Load the inference model and make predictions on new data
loaded_model = tf.keras.models.load_model(ICMP_inference_model)

sample = {
    
    "ip.dst": "192.168.2.15",
    "ip.src": "192.168.2.23",
    "tcp.dstport": 8009,
    "tcp.srcport": 42394,
    "tcp.window_size_value": 1024,
    "tcp.flags": "0x0000",

}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = loaded_model.predict(input_dict)

# Print the prediction probability
print(f"This event has a {100 * predictions[0][0]:.2f}% probability of being of a certain type.")
