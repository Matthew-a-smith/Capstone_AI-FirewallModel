# Import necessary libraries
import os
import tensorflow as tf
import pandas as pd
import keras
from keras.utils import FeatureSpace
from keras.callbacks import Callback
# Set Keras backend to TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"



# Load the dataset
file_url = "test1.csv"
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

# Define FeatureSpace for categorical features
feature_space = FeatureSpace(
    features={
        "Time": FeatureSpace.string_categorical(num_oov_indices=1),
        "Source": FeatureSpace.string_categorical(num_oov_indices=1),
        "Destination": FeatureSpace.string_categorical(num_oov_indices=1),
        "Protocol": FeatureSpace.string_categorical(num_oov_indices=1),
        "Length": FeatureSpace.integer_categorical(num_oov_indices=1),
        "Info": FeatureSpace.string_categorical(num_oov_indices=1),
    },
    crosses=[
        FeatureSpace.cross(feature_names=("Source", "Destination"), crossing_dim=64),
        FeatureSpace.cross(feature_names=("Length", "Info"), crossing_dim=32),
        FeatureSpace.cross(feature_names=("Source", "Length"), crossing_dim=16),
    ],
    output_mode="concat",
)

train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels)

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


class ProbabilityThresholdStop(Callback):
    def __init__(self, threshold):
        super(ProbabilityThresholdStop, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        prob = self.model.predict(preprocessed_val_ds)  # Assuming validation dataset
        max_prob = max(prob)
        if max_prob >= self.threshold:
            print(f"Stopping training because maximum probability ({max_prob}) reached threshold ({self.threshold})")
            self.model.stop_training = True

# Define the threshold
probability_threshold = 0.90  # Set your desired probability threshold here

# Define ProbabilityThresholdStop callback
probability_stop = ProbabilityThresholdStop(probability_threshold)

# Train the model with ProbabilityThresholdStop callback
history = training_model.fit(
    preprocessed_train_ds,
    epochs=40,
    validation_data=preprocessed_val_ds,
    verbose=2,
    callbacks=[probability_stop]  # Pass the ProbabilityThresholdStop callback here
)

# Save the trained and inference models
ICMP_training_model = "AI_Model/Training-Models/NMAP-trained_model.keras"
training_model.save(ICMP_training_model)

ICMP_inference_model = "AI_Model/Validated-Models/NMAP-inference_model.keras"
inference_model.save(ICMP_inference_model)

# Load the inference model and make predictions on new data
loaded_model = tf.keras.models.load_model(ICMP_inference_model)

# Sample input for prediction
sample = {
    "Time": "2024-02-05 14:35:31",
    "Source": "192.168.4.30",
    "Destination": "192.168.4.133",
    "Protocol": "ICMP",
    "Length": 74,
    "Info": ["443  >  40222 [SYN, ACK] Seq=0 Ack=1 Win=65535 Len=0 MSS=1440 WS=256 SACK_PERM"],
}

# Prepare input for prediction
input_dict = {name: tf.constant([value], dtype=tf.string if name in ["Time", "Source", "Destination", "Protocol", "Info"] else tf.int32) for name, value in sample.items()}
predictions = loaded_model.predict(input_dict)

# Print the prediction probability
print(f"This event has a {100 * predictions[0][0]:.2f}% probability of being of a certain type.")

print(inference_model.summary)