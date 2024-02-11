import pandas as pd

# Read the CSV file
df = pd.read_csv('test.csv')

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42)  # Shuffle with a fixed random state for reproducibility

# Convert the 'Time' column to string format with milliseconds included
# Convert the 'Time' column to datetime format
df['Time'] = pd.to_datetime(df['Time'], format='%Y/%j %H:%M:%S.%f')

# Remove milliseconds and format the time as a string
df['Time'] = df['Time'].dt.strftime('%Y/%j %H:%M:%S.%f')

# Convert the 'Time' column to string type explicitly
df['Time'] = df['Time'].astype(str)


# Convert the 'Protocol' column to string type explicitly
df['Protocol'] = df['Protocol'].astype(str)

# Convert 'Source', 'Destination', and 'Length' columns to string type explicitly
df['Source'] = df['Source'].astype(str)
df['Destination'] = df['Destination'].astype(str)
df['Info'] = df['Info'].astype(str)
df['Length'] = df['Length'].astype(str)

# Save the cleaned and shuffled data back to the CSV file, overwriting if necessary
df.to_csv('test3.csv', index=False)
