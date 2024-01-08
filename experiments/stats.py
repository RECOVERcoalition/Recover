import json
import numpy as np

# Specify the path to your JSON file
json_file_path = "result.json"

# Load data from the JSON file
with open(json_file_path, "r") as json_file:
    # Wrap the content in square brackets to form a JSON array
    json_data = "[" + json_file.read().replace("}\n{", "},\n{") + "]"
    
    # Parse the JSON array
    data = json.loads(json_data)

# Extract relevant values
spearman_values = [entry["eval/spearman"] for entry in data]
rsquared_values = [entry["eval/comb_r_squared"] for entry in data]

# Calculate mean and standard deviation
mean_spearman = np.mean(spearman_values)
std_dev_spearman = np.std(spearman_values)

mean_rsquared = np.mean(rsquared_values)
std_dev_rsquared = np.std(rsquared_values)

# Print the results
print(f"Mean eval/rsquared: {round(mean_rsquared, 3)}, Standard Deviation: {round(std_dev_rsquared, 3)}")
print(f"Mean eval/spearman: {round(mean_spearman, 3)}, Standard Deviation: {round(std_dev_spearman, 3)}")

last_entry = data[-1]
mean_r_squared_last = last_entry["mean_r_squared"]
std_r_squared_last = last_entry["std_r_squared"]

mean_spearman_last = last_entry["mean_spearman"]
std_spearman_last = last_entry["std_spearman"]

# Print values for the last entry
print(f"Mean test/rsquared: {round(mean_r_squared_last, 3)}, Standard Deviation: {round(std_r_squared_last, 3)}")
print(f"Mean test/spearman: {round(mean_spearman_last, 3)}, Standard Deviation: {round(std_spearman_last, 3)}")
