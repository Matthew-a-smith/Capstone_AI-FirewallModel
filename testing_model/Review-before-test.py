import pandas as pd

# Load the test data
test_data = pd.read_csv("model_data/Testing_data/ICMP-Test.csv")

# Shuffle the DataFrame
test_data = test_data.sample(frac=1, random_state=42)  # Shuffle with a fixed random state for reproducibility

# Display the first few rows of the test data
print("Sample of test data:")
print(test_data.head())

# Check the shape of the test data
print("\nShape of test data before augmentation:")
print(test_data.shape)

# Check the distribution of the label column
print("\nDistribution of labels before augmentation:")
print(test_data["label"].value_counts())

# Identify rows with label 1
label_1_data = test_data[test_data["label"] == 1].copy()

# Modify the rows with label 1 and add them to the dataset
augmented_data = test_data.copy()  # Create a copy of the original dataset to preserve rows with label 0
for idx, row in label_1_data.iterrows():
    # Modify the row data as needed
    # Here, we can simply clone the row and add it to the dataset
    cloned_row = row.copy()
    augmented_data = augmented_data.append(cloned_row, ignore_index=True)

# Check the shape of the test data after augmentation
print("\nShape of test data after augmentation:")
print(augmented_data.shape)

# Check the distribution of the label column after augmentation
print("\nDistribution of labels after augmentation:")
print(augmented_data["label"].value_counts())

# Check for missing values after augmentation
print("\nMissing values after augmentation:")
print(augmented_data.isnull().sum())
