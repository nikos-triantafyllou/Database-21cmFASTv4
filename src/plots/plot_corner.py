import json
import numpy as np
import matplotlib.pyplot as plt
import corner

# Load the data from the JSON file
input_file = '../LHS_SAMPLES.json'
with open(input_file, 'r') as f:
    data = json.load(f)

# Extract the parameter names (assuming they are consistent across samples)
first_key = list(data.keys())[0]
param_names = list(data[first_key].keys())

# Extract the parameter values from each sample
samples = []
for key in data:
    sample = [data[key][param] for param in param_names]
    samples.append(sample)

# Convert the samples to a NumPy array for plotting
samples = np.array(samples)

# Since 'L_X_MINI' is identical to 'L_X', we remove it from the plot to avoid redundancy
param_names_for_plot = [name for name in param_names if name != 'L_X_MINI']
samples_for_plot = samples[:, [i for i, name in enumerate(param_names) if name != 'L_X_MINI']]

# Create the corner plot
fig = corner.corner(samples_for_plot, labels=param_names_for_plot, show_titles=False, title_fmt=".2f", plot_contours=False,
                    hist2d_kwargs={"plot_datapoints": True, "plot_density": False, "plot_contours": False} )

# Save the plot as a PDF
output_pdf = 'corner_plot.pdf'
fig.savefig(output_pdf)

# Show the plot
plt.show()

print(f"Corner plot saved as {output_pdf}")
