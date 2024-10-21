import json
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import corner


# # Function to generate varying parameters
# def generate_varying_params():
#     varying_params = {
#         "random_seed": random.randint(0, 100000),
#         "SIGMA_8": np.random.normal(0.811, 0.006),
#         "L_X": random.uniform(38, 42),
#         "L_X_MINI": 'to be set as the same with L_X',
#         "NU_X_THRESH": random.uniform(100, 1500),
#         "F_STAR10": random.uniform(-3, 0),
#         "ALPHA_STAR": random.uniform(-0.5, 1),
#         "F_ESC10": random.uniform(-3, 0),
#         "ALPHA_ESC": random.uniform(-1, 0.5),
#         "t_STAR": random.uniform(0, 1),
#         "SIGMA_SFR_LIM": random.uniform(0.01, 1.2),
#         "M_TURN": random.uniform(5, 10)
#     }
#     # Keep L_X and L_X_MINI consistent
#     Lx = varying_params['L_X']
#     varying_params['L_X_MINI'] = Lx
    
#     return varying_params


from scipy.stats import qmc
sampler = qmc.LatinHypercube(d=11)
sample = sampler.random(n=1000)

l_bounds = [0.8092,   #SIGMA_8
            38,       #L_X
            38,       #L_X_MINI
            100,      #NU_X_THRESH
            -3,       #F_STAR10
            -0.5,     #ALPHA_STAR
            -3,       #F_ESC10
            -1,       #ALPHA_ESC
            0,        #t_STAR
            0,        #SIGMA_SFR_LIM
            5,        #M_TURN
            ]    
u_bounds = [0.8128,   #SIGMA_8
            42,       #L_X
            42,       #L_X_MINI
            1500,     #NU_X_THRESH
            0,        #F_STAR10
            1,        #ALPHA_STAR
            0,        #F_ESC10
            0.5,      #ALPHA_ESC
            1,        #t_STAR
            1.2,      #SIGMA_SFR_LIM
            10,       #M_TURN
            ]    

sample_scaled = qmc.scale(sample, l_bounds, u_bounds)



param_names = [
    "SIGMA_8", "L_X", "L_X_MINI", "NU_X_THRESH", "F_STAR10", "ALPHA_STAR",
    "F_ESC10", "ALPHA_ESC", "t_STAR", "SIGMA_SFR_LIM", "M_TURN"
]


# Prepare the data to be written in JSON format
all_runs = []
for s in sample_scaled:
    sample_dict = {param_names[i]: s[i] for i in range(len(param_names))}
    # Ensure consistency for L_X and L_X_MINI
    sample_dict['L_X_MINI'] = sample_dict['L_X']
    all_runs.append(sample_dict)

# Save the results to a JSON file
output_file = 'LHS_SAMPLES.json'
with open(output_file, 'w') as f:
    json.dump({i: run for i, run in enumerate(all_runs)}, f, indent=4)

print(f"Latin Hypercube samples saved to {output_file}")


