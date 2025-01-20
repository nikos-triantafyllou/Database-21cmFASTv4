# #IMPORTS----------------------------------------------------------------------

print("Initiating python script...", flush=True)

# %matplotlib inline
import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
import glob
import os, sys
# print('Conda environment:', os.environ["CONDA_PREFIX"],flush=True)
print('Python version:', sys.version, flush=True)
from datetime import datetime

# We change the default level of the logger so that
# we can see what's happening with caching.
import logging, sys, os
logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)

version_info = sys.version_info
python_version=f'{version_info.major}.{version_info.minor}.{version_info.micro}'

print('Importing 21cmFAST...', flush=True)
import py21cmfast as p21c
print('Imported 21cmFAST', flush=True)

# For plotting the cubes, we use the plotting submodule:
from py21cmfast import plotting

# For interacting with the cache
from py21cmfast import cache_tools

import py21cmfast_tools as p21c_tools
print(f"Using 21cmFAST version {p21c.__version__}",flush=True)

# if not os.path.exists('_jup_cache'):
#     os.mkdir('_jup_cache')
# p21c.config['direc'] = '_jup_cache'
# # cache_tools.clear_cache(direc="_cache")

from astropy.cosmology import Planck15
import astropy.constants as const

print('Fixing Cosmology...', flush=True)

# Fix Planck cosmology for 21cmFAST
Planck18 = Planck15.clone(
    Om0=(0.02242 + 0.11933) / 0.6766**2,
    Ob0=0.02242 / 0.6766**2,
    H0=67.66,
    name="Planck18",
)

_a_B_c2 = (4 * const.sigma_sb / const.c**3).cgs.value
def Tcmb0(Ogamma0):
        """Tcmb; the temperature of the CMB at z=0."""
        return pow ( Ogamma0 *  Planck18.critical_density0.value / _a_B_c2 , 1/4)

Planck18 = Planck15.clone(
    Om0=(0.02242 + 0.11933) / 0.6766**2,
    Ob0=0.02242 / 0.6766**2,
    H0=67.66,
    name="Planck18",
    Neff = 0,
    Tcmb0=Tcmb0(8.600000001024455e-05)
)

print('Determining Simulation parameters...', flush=True)
# SIMULATION PARAMETERS
global_params={}



# Fiducial parameters:

cosmo_params = {
        "SIGMA_8": 0.8102,       # ----------------------------------database---------------------------------------
        "hlittle": Planck18.h,
        "OMm": Planck18.Om0,
        "OMb": Planck18.Ob0,
        "POWER_INDEX": 0.9665,
    }


user_params = {
        "BOX_LEN": 300.0,
        "DIM": None,
        "HII_DIM": 200,
        "NON_CUBIC_FACTOR": 1.0,
        "USE_FFTW_WISDOM": False,
        "HMF": 1,
        "USE_RELATIVE_VELOCITIES": True, #------for mini halos---------------
        "POWER_SPECTRUM": 5,             #------------from 0, since I am using relative velocities----------
        "N_THREADS": 9, # -------------------change this based on the cluster-------------------------------
        "PERTURB_ON_HIGH_RES": False,
        "NO_RNG": False,
        "USE_INTERPOLATION_TABLES": True, #default None
        "INTEGRATION_METHOD_ATOMIC": 1,
        "INTEGRATION_METHOD_MINI": 1,
        "USE_2LPT": True,
        "MINIMIZE_MEMORY": False,
        "STOC_MINIMUM_Z": None,
        "KEEP_3D_VELOCITIES": False,
        "SAMPLER_MIN_MASS": 1e8,
        "SAMPLER_BUFFER_FACTOR": 2.0,
        "MAXHALO_FACTOR": 2.0,
        "N_COND_INTERP": 200,
        "N_PROB_INTERP": 400,
        "MIN_LOGPROB": -12,
        "SAMPLE_METHOD": 0,
        "AVG_BELOW_SAMPLER": True,
        "HALOMASS_CORRECTION": 0.9,
    }




flag_options = {
        "USE_HALO_FIELD": True,
        "USE_MINI_HALOS": True, # mini halos changed this 
        "USE_CMB_HEATING": True,
        "USE_LYA_HEATING": True,
        "USE_MASS_DEPENDENT_ZETA": True,
        "SUBCELL_RSD": False,
        "APPLY_RSDS": True,
        "INHOMO_RECO": True, # changed for mini halos 
        "USE_TS_FLUCT": True, #changed this 
        "M_MIN_in_Mass": True, #changed this 
        "FIX_VCB_AVG": False,
        "HALO_STOCHASTICITY": True,
        "USE_EXP_FILTER": True,
        "FIXED_HALO_GRIDS": False,
        "CELL_RECOMB": True,
        "PHOTON_CONS_TYPE": 0,  # Should these all be boolean?
        "USE_UPPER_STELLAR_TURNOVER": True,
    }



astro_params = {
    "HII_EFF_FACTOR": 30.0,
    "F_STAR10": -1.3,   # ----------------------------------database---------------------------------------
    "F_STAR7_MINI": None,
    "ALPHA_STAR": 0.5,  # ----------------------------------database---------------------------------------
    "ALPHA_STAR_MINI": None,
    "F_ESC10": -1.0,     # ----------------------------------database---------------------------------------
    "F_ESC7_MINI": -2.0, 
    "ALPHA_ESC": -0.5,   # ----------------------------------database---------------------------------------
    "M_TURN": None,       # ----------------------------------database---------------------------------------
    "R_BUBBLE_MAX": None,
    "ION_Tvir_MIN": 4.69897,
    "L_X": 40.5,  # Kaur+22  # ----------------------------------database---------------------------------------
    "L_X_MINI": 40.5,
    "NU_X_THRESH": 500.0,  # E0 ----------------------------------database---------------------------------------
    "X_RAY_SPEC_INDEX": 1.0,
    "X_RAY_Tvir_MIN": None,
    "F_H2_SHIELD": 0.0,
    "t_STAR": 0.5,          # ----------------------------------database---------------------------------------
    "N_RSD_STEPS": 20,
    "A_LW": 2.00,
    "BETA_LW": 0.6,
    "A_VCB": 1.0,
    "BETA_VCB": 1.8,
    "UPPER_STELLAR_TURNOVER_MASS": 11.447,  # 2.8e11
    "UPPER_STELLAR_TURNOVER_INDEX": -0.6,
    # Nikolic et al. 2024 lognormal scatter parameters
    "SIGMA_STAR": 0.25,
    "SIGMA_LX": 0.5,
    "SIGMA_SFR_LIM": 0.19,   # ----------------------------------database---------------------------------------
    "SIGMA_SFR_INDEX": -0.12,
    # Self-Correlations based on cursory examination of Astrid-ES data (Davies et al 2023)
    "CORR_STAR": 0.5,
    "CORR_SFR": 0.2,
    "CORR_LX": 0.2,  # NOTE (Jdavies): It's difficult to know what this should be, ASTRID doesn't have the xrays and I don't know which hydros do
}


global_params={}


random_seed = 23


varying_params = { # Doesn't matter
                    "random_seed": 23,
                   # Priors from planck
                   "SIGMA_8": 0.8102,
                   
                   # Uniform
                   "L_X": 40.0,
                   "NU_X_THRESH": 500.0,
                   
                   # Park+2019
                   "F_STAR10": -1.3, 
                    # Park?
                   "ALPHA_STAR": 0.5, 
                   # Park+2019
                   "F_ESC10": -1.0,
                   # Park?
                   "ALPHA_ESC": -0.5,
                   
                   "t_STAR": 0.5,
                   
                   "SIGMA_SFR_LIM": 0.6,
                   
                  "M_TURN":  None , 
#                   "T_eff":     Andrei slack:  for the future->ignore 
                  }





print('Determining cache directory...', flush=True)


# Create cache directory in the same directory as the .py file
if not os.path.exists('_cache'):
    os.mkdir('_cache')
p21c.config['direc'] = '_cache'
# cache_tools.clear_cache(direc="_cache")
cache_path = str( os.path.dirname(os.path.realpath(__file__)) ) + '/_cache/'




random_seed = varying_params['random_seed']
cosmo_params['SIGMA_8'] = varying_params['SIGMA_8']

for astro_key in list(varying_params.keys())[2:]:
    astro_params[astro_key] = varying_params[astro_key]



print('Run 21cmFAST...', flush=True)


redshift = 5
# Create the lightcone
lightcone = p21c.run_lightcone(
    redshift = redshift,
#     max_redshift = 6.0,
    user_params = user_params,
    cosmo_params = cosmo_params,
    astro_params= astro_params,
    flag_options= flag_options,
    lightcone_quantities=("brightness_temp", 'Ts_box','xH_box',"density","Tk_box", "Gamma12_box", "x_e_box", "n_ion",'halo_mass','halo_sfr','halo_stars'
                         ,'halo_sfr_mini','halo_stars_mini'
                         ),
    random_seed=random_seed,
    global_quantities=("brightness_temp", 'xH_box','Tk_box'),
#     write=False
)

print('Simulation successful', flush=True)

print('Saving...', flush=True)

# # lightcone.save(fname = "test_lightcone.h5")


from d_utils import create_group_with_attributes, search_files 


# savedir = '/home/ntriantafyllou/projects/21cmFASTv4-development/'
filename = 'test_simple_run.h5'

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
    
    
    #UV_LF
    # z_uv=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # LF_arr=p21c.compute_luminosity_function(
    #     redshifts = z_uv,
    #     user_params=user_params,
    #     cosmo_params=cosmo_params,
    #     astro_params=astro_params,
    #     flag_options=flag_options,
    #     nbins=100,
    #     mturnovers=None,
    #     mturnovers_mini=None,
    #     component=0,
    # )    
    # data_UV_LF = group_coeval_data.create_dataset('UV_LF',              data= LF_arr)
    # data_UV_LF.attrs['z_uv'] =  z_uv
    
    # EOR History
    data_EoR_history = group_coeval_data.create_dataset('EoR_history',  data = lightcone.global_quantities['xH_box'])
    
    
    # Tau_e
    tau_e = p21c.compute_tau(redshifts = lightcone.node_redshifts[::-1],  # reverse redshifts so that they are in acccending order 
                             global_xHI = lightcone.global_quantities['xH_box'][::-1], # reverse xHI  so that they are in acccending order like the redshifts
                             user_params=user_params, 
                             cosmo_params=cosmo_params)
    data_tau_e = group_coeval_data.create_dataset('tau_e',              data = tau_e)
    
    # Tb
    data_Tb_z_global = group_coeval_data.create_dataset('Tb_z_global',  data = lightcone.global_quantities['brightness_temp'])

    # Tk
    data_Tk_z_global = group_coeval_data.create_dataset('Tk_z_global',  data = lightcone.global_quantities['Tk_box'])
    
    # 21cm PS
    data_21cm_PS = group_coeval_data.create_dataset('21cm_PS',          dtype=np.float64)
    
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
    data_density = group_lightcones.create_dataset('Density',                  data = lightcone.density)
    data_X_HI = group_lightcones.create_dataset('X_HI',                        data = lightcone.xH_box)
    data_T_kin = group_lightcones.create_dataset('T_kin',                      data = lightcone.Tk_box)
    data_T_spin = group_lightcones.create_dataset('T_spin',                    data = lightcone.Ts_box)
    data_Gamma = group_lightcones.create_dataset('Gamma',                      data = lightcone.Gamma12_box)
    data_ion_emissivity = group_lightcones.create_dataset('Ion_Emissivity',    data = lightcone.n_ion)
    data_Xray_emissivity = group_lightcones.create_dataset('Xray_emissivity',  data = lightcone.x_e_box)
    
    #     # Create a dataset and save data1
    #     hdf.create_dataset('dataset1', data=data1)
    #     group.create_dataset('dataset1', data=data1)

print(f"First data has been saved to {filename}", flush=True)

# Lazy RAM cleaning 
# lightcone = 0 



print("Now adding HMFs...", flush=True)


with h5py.File(filename, 'r') as hdf:
    node_redshifts = hdf['coeval_data'].attrs['node_redshifts']
    box_len = hdf['simulation_parameters']['user_params'].attrs["BOX_LEN"]
    

# Calculate the HMF from the mass array in the PerturbHaloField file
pattern = 'PerturbHaloField*' 
found_files = search_files(cache_path, pattern)
    

BoxSize = box_len # Mpc/h
min_mass = 1e8 #minimum mass in Msun/h
max_mass = 1e14 #maximum mass in Msun/h
bins     = 40   #number of bins in the HMF
bins_mass = np.logspace(np.log10(min_mass), np.log10(max_mass), bins+1)
mass_mean = 10**(0.5*(np.log10(bins_mass[1:])+np.log10(bins_mass[:-1])))
dM        = bins_mass[1:] - bins_mass[:-1]

HMF_dict={}
for cache_file in found_files:  
    with h5py.File(f'{cache_path}{cache_file}', 'r') as hdf:
        hmass_arr = hdf['PerturbHaloField']['halo_masses'][:]
        HMF_dict[dict(hdf.attrs)['redshift']] = mass_mean * np.histogram(hmass_arr, bins=bins_mass)[0]/(dM*BoxSize**3)

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

    data_HMF_z.attrs['info']= 'dn/dlnM[?], shape: ( len(node_redshifts), len(hist_mass_mean) )'


print("Added the HMFs", flush=True)

from d_utils import explore_hdf5
print("Description of the file structure:", flush=True)
explore_hdf5(filename)


