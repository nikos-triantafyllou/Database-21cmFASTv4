import json
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Function to generate varying parameters
def generate_varying_params():
    varying_params = {
        "random_seed": random.randint(0, 100000),
        "SIGMA_8": np.random.normal(0.811, 0.006),
        "L_X": random.uniform(38, 42),
        "L_X_MINI": random.uniform(38, 42),
        "NU_X_THRESH": random.uniform(100, 1500),
        "F_STAR10": random.uniform(-3, 0),
        "ALPHA_STAR": random.uniform(-0.5, 1),
        "F_ESC10": random.uniform(-3, 0),
        "ALPHA_ESC": random.uniform(-1, 0.5),
        "t_STAR": random.uniform(0, 1),
        "SIGMA_SFR_LIM": random.uniform(0.01, 1.2),
        "M_TURN": random.uniform(5, 10)
    }
    # Keep L_X and L_X_MINI consistent
    Lx = varying_params['L_X']
    varying_params['L_X_MINI'] = Lx
    
    return varying_params

# Function to run the generation in parallel
def generate_all_params_parallel(num_runs):
    all_results = []
    with ProcessPoolExecutor() as executor:
        # Use submit instead of map
        futures = [executor.submit(generate_varying_params) for _ in range(num_runs)]
        
        for future in as_completed(futures):
            all_results.append(future.result())
    
    return all_results

# Number of runs
num_runs = 10000  # Adjust the number of runs if needed

# Parallelize the generation
all_runs = generate_all_params_parallel(num_runs)

# Save the results to a JSON file
output_file = 'VARYING_PARAMETERS.json'
with open(output_file, 'w') as f:
    json.dump({i: run for i, run in enumerate(all_runs)}, f, indent=4)

print(f"Parameter sets saved to {output_file}")
