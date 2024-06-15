import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# read in raw data
file_path = 'wdbc.data'
df = pd.read_csv(file_path, header=None)

# classes: B (benign) = 1, M (malignant) = 2
labels = df.iloc[:, 1].map({'B': 1, 'M': 2}).astype(float) 
features = df.iloc[:, 2:]
features = features.applymap(lambda x: float(x) if is_float(x) else np.nan)

# Z-normalization 
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Combine labels and normalized features
data_combined = np.column_stack((labels, features_normalized))
np.savetxt("normalized_raw.txt", data_combined, delimiter = "  ")

# truncate values to 7 decimal places (like in given datasets)
input_file = 'normalized_raw.txt'
with open(input_file, 'r') as infile:
    lines = infile.readlines()

truncated_lines = []
for line in lines:
    values = line.split()
    truncated_values = [f"{float(values[0]):.7e}"] # class label
   
    # average feature values across the 3 nuclei
    for i in range(1,11):
        value = (float(values[i]) + float(values[i+10]) + float(values[i+20])) / 3
        truncated_value = float(value)
        truncated_values.append(f"{truncated_value:.7e}")

    truncated_line = ' '.join(truncated_values)
    truncated_lines.append(truncated_line)

output_file = 'normalized_average.txt'    # write values to output file
with open(output_file, 'w') as outfile:
    for line in truncated_lines:
        outfile.write(f"{line}\n")

