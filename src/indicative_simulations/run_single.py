TEST_WITH_SMALL_RUN = False # If set to 'True' 21cmFast will run a 60Mpc box with 60x60x60 resolution up until z=34 

import json
import argparse
import time
import numpy as np

import py21cmfast as p21c
from py21cmfast import plotting
from py21cmfast import cache_tools
import py21cmfast_tools as p21c_tools

from astropy.cosmology import Planck15

import os, sys
import h5py
from datetime import datetime
version_info = sys.version_info
python_version=f'{version_info.major}.{version_info.minor}.{version_info.micro}'

from d_utils import create_group_with_attributes, search_files, explore_hdf5

from astropy import units as un

'''========================================= Determine directories =================================='''

parser = argparse.ArgumentParser()
parser.add_argument("--savedir", type = str, default = "/leonardo_scratch/large/userexternal/ntriant1/database/final/indicative_simulations/save/")
parser.add_argument("--logdir", type = str, default = "/leonardo_scratch/large/userexternal/ntriant1/database/final/indicative_simulations/logs/")
parser.add_argument("--cache_dir", type=str, default="/leonardo_scratch/large/userexternal/ntriant1/database/final/indicative_simulations/_cache/")
parser.add_argument("--counter", type = int, default = 0)
parser.add_argument("--threads", type=int, default=4)
inputs = parser.parse_args()



counter = inputs.counter
savedir = inputs.savedir
threads = inputs.threads
print(f'Running with counter: {counter}', flush=True)

cache_path = f"/leonardo_scratch/large/userexternal/ntriant1/database/final/indicative_simulations/_cache/_cache_id_{counter}/"             
if not os.path.exists(cache_path):                                                              
    os.mkdir(cache_path)                                                                        
p21c.config['direc'] = cache_path                                                               
cache_tools.clear_cache(direc=cache_path)  

filename = savedir+f'id_{counter}.h5'   

'''======================================== LOAD PARAMETERS ========================================='''

# Fiducial parameters from FIDUCIAL_PARAMETERS.json
file_path = "./FIDUCIAL_PARAMETERS.json"
with open(file_path, "r") as json_file:
    data = json.load(json_file)
cosmo_params = data["cosmo_params"]
user_params = data["user_params"]
flag_options = data["flag_options"]
astro_params = data["astro_params"]
global_params=  {}

# Setup default values indicated as strings in the "FIDUCIAL_PARAMETERS.json" file
Planck18 = Planck15.clone(
    Om0=(0.02242 + 0.11933) / 0.6766**2,
    Ob0=0.02242 / 0.6766**2,
    H0=67.66,
    name="Planck18",
)

cosmo_params = {
        "SIGMA_8": 0.8102,       # ----------------------------------database---------------------------------------
        "hlittle": Planck18.h,
        "OMm": Planck18.Om0,
        "OMb": Planck18.Ob0,
        "POWER_INDEX": 0.9665,
    }

user_params['N_THREADS'] = threads




# Change the values related to the database
# file_path = "./no_stoc_sfr_SAMPLES.json"
file_path = "./PLUS_MINUS_PARAMS.json"

with open(file_path, "r") as json_file:
    parameter_dict = json.load(json_file)
parameter_dict_keys = list(parameter_dict.keys())
# varying_params = parameter_dict[parameter_dict_keys[counter-1]]
varying_params = parameter_dict[str(counter)]

random_seed = varying_params['random_seed']
cosmo_params['SIGMA_8'] = varying_params['SIGMA_8']
for astro_key in list(varying_params.keys())[2:]:
    astro_params[astro_key] = varying_params[astro_key]

# Change related mini values
astro_params['F_ESC7_MINI'] = astro_params['F_ESC10']-1                                   # ----------------------------------database---------------------------------------
astro_params['F_STAR7_MINI'] = astro_params['F_STAR10'] - 3 * astro_params['ALPHA_STAR']  # -3*alpha since 1e7/1e10 = 1e-3----database---------------------------------------
astro_params['ALPHA_STAR_MINI'] = astro_params['ALPHA_STAR']                              # ----------------------------------database---------------------------------------
astro_params['L_X_MINI'] = astro_params['L_X']                                            # ----------------------------------database---------------------------------------


print(f'Running with values: {varying_params}', flush=True)
print(cosmo_params)





'''======================================== RUN 21cmFAST =========================================='''

print('Run 21cmFAST...', flush=True)

redshift = 5

if TEST_WITH_SMALL_RUN: 
    user_params["BOX_LEN"]=60.0
    user_params["HII_DIM"]=60
    redshift=34

print(user_params)


# Run 21cmFAST--------------------------------------------------------------------------------------------------------------------------------------

print("These are the new user params defaults:", p21c.UserParams.new(dict()))
print("These are the new flag options defaults:", p21c.FlagOptions.new(dict()))
cosmo_params_struct = p21c.CosmoParams.new(cosmo_params)
flag_options_struct = p21c.FlagOptions.new(flag_options)
astro_params_struct = p21c.AstroParams.new(astro_params)
user_params_struct = p21c.UserParams.new(user_params)


min_redshift=redshift
max_redshift=35.0


node_redshifts = p21c.get_logspaced_redshifts(min_redshift  = min_redshift,
                             max_redshift  = max_redshift,
                             z_step_factor = user_params['ZPRIME_STEP_FACTOR'])



input_params_struct = p21c.InputParameters(
                cosmo_params=cosmo_params_struct,
                astro_params=astro_params_struct,
                user_params=user_params_struct,
                flag_options=flag_options_struct,
                random_seed=random_seed,
                node_redshifts = node_redshifts,
            )


#Generate a LightConer instance, which defines the dimensions, angles and quantities desired
lightcone_quantities=("brightness_temp", 
                          'Ts_box',
                          'xH_box',
                          "density",
                          "Tk_box", 
                          "temp_kinetic_all_gas",
                          "Gamma12_box", 
                          "x_e_box", 
                          "n_ion",
                          'halo_mass','halo_sfr','halo_stars',
                         'halo_sfr_mini','halo_stars_mini',
                         'velocity_z'
                         )

global_quantities=("brightness_temp", 
                       'xH_box', 
                    #    'Tk_box', 
                       "temp_kinetic_all_gas")






lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=redshift,
        max_redshift=35.0,
        resolution=user_params['BOX_LEN'] / user_params['HII_DIM'] * un.Mpc,
        quantities=lightcone_quantities,
)
iz, z, coev, lightcone = p21c.exhaust_lightcone(
        lightconer=lcn,
        inputs = input_params_struct,
        regenerate = True,
        global_quantities=global_quantities,
)


#------------------------------------------------------------------------------------------------------------------------------------


print('Simulation successful', flush=True)




'''======================================== SAVE RESULTS 1 =========================================='''
 
print('Saving...', flush=True)

# Open an HDF5 file in write mode (creates the file if it doesn't exist)
with h5py.File(filename, 'w') as hdf:
    
    hdf.attrs['21cmFAST_version'] = p21c.__version__
    hdf.attrs['python_version'] = python_version
    hdf.attrs['random_seed'] = random_seed
    hdf.attrs['redshift'] = redshift
    hdf.attrs['creation date/time'] = str(datetime.now())
    
    # Simulation Params-----------------------------------------------------------------------
    group_simulation_parameters = hdf.create_group('simulation_parameters')
    
    # Create subgroups and add attributes
    create_group_with_attributes(group_simulation_parameters, 'user_params', user_params)
    create_group_with_attributes(group_simulation_parameters, 'cosmo_params', cosmo_params)
    create_group_with_attributes(group_simulation_parameters, 'astro_params', astro_params)
    create_group_with_attributes(group_simulation_parameters, 'flag_options', flag_options)
    create_group_with_attributes(group_simulation_parameters, 'global_params', global_params)
    create_group_with_attributes(group_simulation_parameters, 'varying_params', varying_params)
    
    
    
    # coeval_data---------------------------------------------------------------------------------
    group_coeval_data = hdf.create_group('coeval_data')
    group_coeval_data.attrs['node_redshifts'] = lightcone.node_redshifts    
    
    # EOR History
    data_x_HI = group_coeval_data.create_dataset('x_HI',  data = lightcone.global_quantities['xH_box'])
    
    
    # Tau_e
    tau_e = p21c.compute_tau(redshifts = lightcone.node_redshifts[::-1],  # reverse redshifts so that they are in ascending order 
                             global_xHI = lightcone.global_quantities['xH_box'][::-1], # reverse xHI  so that they are in ascending order like the redshifts
                             inputs = input_params_struct)
    data_tau_e = group_coeval_data.create_dataset('tau_e',              data = tau_e)
    
    # Tb
    data_brightness_temp = group_coeval_data.create_dataset('brightness_temp',  data = lightcone.global_quantities['brightness_temp'])

    # Tk
    # data_Tk_z_global = group_coeval_data.create_dataset('Tk_z_global',  data = lightcone.global_quantities['Tk_box'])
    
    # Temp_kinetic_all_gas
    data_T_kin_all_gas = group_coeval_data.create_dataset('T_kin_all_gas',  data = lightcone.global_quantities['temp_kinetic_all_gas'])

    log10_mturnovers = group_coeval_data.create_dataset('log10_mturnovers',  data = lightcone.log10_mturnovers)

    log10_mturnovers_mini = group_coeval_data.create_dataset('log10_mturnovers_mini',  data = lightcone.log10_mturnovers_mini)


    # 21cm PS
    # data_21cm_PS = group_coeval_data.create_dataset('21cm_PS',          dtype=np.float64)
    
    # HMF vs z: for this I will reopen the file only when I calculate the relations so that I dont loose RAM 
    data_HMF_z = group_coeval_data.create_dataset('HMF_z',              dtype=np.float64)
    
    
    # Lightcones------------------------------------------------------------
    group_lightcones = hdf.create_group('lightcones')
    group_lightcones.attrs['lightcone_redshifts'] = lightcone.lightcone_redshifts     
    
    
    data_brightness_temp = group_lightcones.create_dataset('brightness_temp',  data = lightcone.brightness_temp)
    data_SFR = group_lightcones.create_dataset('SFR',                          data = lightcone.halo_sfr)
    # mini 
    data_SFR_mini = group_lightcones.create_dataset('SFR_mini',                data = lightcone.halo_sfr_mini)
    data_M_halo = group_lightcones.create_dataset('M_halo',                    data = lightcone.halo_mass)
    data_M_star = group_lightcones.create_dataset('M_star',                    data = lightcone.halo_stars)
    # mini
    data_M_star_mini = group_lightcones.create_dataset('M_star_mini',          data = lightcone.halo_stars_mini)
    data_density = group_lightcones.create_dataset('density',                  data = lightcone.density)
    data_X_HI = group_lightcones.create_dataset('x_HI',                        data = lightcone.xH_box)
    # data_T_kin = group_lightcones.create_dataset('T_kin',                      data = lightcone.Tk_box)
    data_T_kin_all_gas = group_lightcones.create_dataset('T_kin_all_gas',      data = lightcone.temp_kinetic_all_gas)
    data_T_spin = group_lightcones.create_dataset('T_spin',                    data = lightcone.Ts_box)
    data_Gamma = group_lightcones.create_dataset('Gamma',                      data = lightcone.Gamma12_box)
    data_ion_emissivity = group_lightcones.create_dataset('ion_emissivity',    data = lightcone.n_ion)
    data_Xray_emissivity = group_lightcones.create_dataset('xray_emissivity',  data = lightcone.x_e_box)
    data_velocity_z = group_lightcones.create_dataset('velocity_z',  data = lightcone.velocity_z)

    


    group_halo_data = hdf.create_group('halo_data')


print(f"First data has been saved to {filename}", flush=True)







'''====================================== SAVE RESULTS 2 (add HMF from cache)=========================================='''



print("Now adding HMFs...", flush=True)


with h5py.File(filename, 'r') as hdf:
    node_redshifts = hdf['coeval_data'].attrs['node_redshifts']
    box_len = hdf['simulation_parameters']['user_params'].attrs["BOX_LEN"]
    

# Calculate the HMF from the mass array in the PerturbHaloField file
pattern = 'PerturbHaloField*' 
found_files = search_files(cache_path, pattern)
    

BoxSize = box_len # Mpc
min_mass = 1e8 #minimum mass in Msun
max_mass = 1e14 #maximum mass in Msun
bins     = 40   #number of bins in the HMF
bins_mass = np.logspace(np.log10(min_mass), np.log10(max_mass), bins+1)
mass_mean = 10**(0.5*(np.log10(bins_mass[1:])+np.log10(bins_mass[:-1])))
dM        = bins_mass[1:] - bins_mass[:-1]

HMF_dict={}
coupled_names_dict={}
for cache_file in found_files:  
    with h5py.File(f'{cache_path}{cache_file}', 'r') as hdf:
        hmass_arr = hdf['PerturbHaloField']['halo_masses'][:]
        HMF_dict[dict(hdf.attrs)['redshift']] = mass_mean * np.histogram(hmass_arr, bins=bins_mass)[0]/(dM*BoxSize**3)

        # Added line to couple redshifts and names of files 
        coupled_names_dict[dict(hdf.attrs)['redshift']] = f'{cache_file}'

print('coupled dictionary:',coupled_names_dict, flush=True)     

HMF_arr = np.zeros((len(node_redshifts), len(mass_mean)))
ii = 0
for node_redshift in node_redshifts:
    HMF_arr[ii] = HMF_dict[node_redshift]
    ii+=1
    
    
# Open an HDF5 file again and write the HMF vs z
with h5py.File(filename, 'a') as hdf:
    
    # Create a new dataset within the group
    group = hdf['coeval_data']
    dset_name = 'HMF_z'
    if dset_name in group:
        print(f'Warning: rewriting {dset_name} data')
        del group[dset_name]  # Optional: delete the existing dataset if it exists
        
    data_HMF_z = group.create_dataset(dset_name,  data=HMF_arr)
    data_HMF_z.attrs['hist_mass_mean']=mass_mean

    data_HMF_z.attrs['info']= 'dn/dlnM[Mpc^{-3}] vs M[M_\odot], shape: ( len(node_redshifts), len(hist_mass_mean) )'


print("Added the HMFs", flush=True)


print("Description of the file structure:", flush=True)
explore_hdf5(filename)


'''====================================== SAVE RESULTS 3 (add PS from cache???)=========================================='''





'''====================================== Adding halos =========================================='''

print('Now adding halos...', flush=True)


# Add section to the hdf5 file
# halo data---------------------------------------------------------------------------------

# Save only ceertain redshifts around 5.5, 6, 6.5, 7, 8, 9
# The names of the dictionary are ints and can be found in the easy_access_info folder of this repository (because tthey are nor exactly 5.5,6.0 etc)

filenames_in_order = []
if TEST_WITH_SMALL_RUN==True:
    for name_of_node_z in node_redshifts:
    # for name_of_node_z in [34.69999933242798,34.0]:
        filenames_in_order.append(coupled_names_dict[name_of_node_z])
elif TEST_WITH_SMALL_RUN==False:
    for name_of_node_z in [9.04050868612415, 8.09399806338753, 7.075210029944779, 6.460245850367914, 6.029956286013594, 5.49459296]:
        print('found node_redshift', flush=True)
        filenames_in_order.append(coupled_names_dict[name_of_node_z])

with h5py.File(filename, 'a') as hdf_out:
    node_redshifts = hdf_out['coeval_data'].attrs['node_redshifts']
    group_halo_data = hdf_out['halo_data']

    prog_i=0
    for cache_file in filenames_in_order:  # so different redshifts
        print(prog_i, flush=True)
        # Read the specific cache perturb file of a certain redshift
        with h5py.File(f'{cache_path}{cache_file}', 'r') as hdf_in:
            print('read the file',flush=True)
            hmass_arr = hdf_in['PerturbHaloField']['halo_masses'][:]
            print('shape=',hmass_arr.shape,flush=True)
            redshift = dict(hdf_in.attrs)['redshift']

            print('selecting...',flush=True)
            sel = hmass_arr > 10**(10)
            # indices95 = np.where(hmass_arr > 10**(9.5) )[0]
            print('writing in ram...',flush=True)
            halo_masses = hdf_in['PerturbHaloField']['halo_masses'][:][sel]
            halo_coords = hdf_in['PerturbHaloField']['halo_coords'][:][sel,:]
            sfr_rng     = hdf_in['PerturbHaloField']['sfr_rng'][:][sel]
            star_rng    = hdf_in['PerturbHaloField']['star_rng'][:][sel]
        
        print('writing...', flush=True)
        # Open an HDF5 file again and write the halo data
        group_halo_data_redshift = group_halo_data.create_group(f'{redshift}')
        data_halo_masses = group_halo_data_redshift.create_dataset('halo_masses',  data = halo_masses)
        data_halo_coords = group_halo_data_redshift.create_dataset('halo_coords',  data = halo_coords)
        data_sfr_rng     = group_halo_data_redshift.create_dataset('sfr_rng',      data = sfr_rng)
        data_star_rng    = group_halo_data_redshift.create_dataset('star_rng',     data = star_rng)

        prog_i+=1


explore_hdf5(filename)

print('FINISHED', flush=True)




    