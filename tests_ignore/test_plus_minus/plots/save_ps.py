import matplotlib.pyplot as plt
import h5py
import numpy as np

from astropy.cosmology import Planck15
import astropy.constants as const

import d_utils
from py21cmfast.cache_tools import readbox


import py21cmfast as p21c
from py21cmfast.c_21cmfast import ffi, lib

import logging, sys, os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from matplotlib import pyplot as plt
from timeit import default_timer as timer
from datetime import timedelta

import py21cmfast_tools as p21c_tools

from d_plotting import get_contour_stuff
from d_plotting import get_mean_std_stuff

import argparse
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--savedir", type = str, default = "/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/save6/")
parser.add_argument("--logdir", type = str, default = "/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/")
parser.add_argument("--workers", type = int, default = 1)
parser.add_argument("--seed", type = int, default = 22)
parser.add_argument("--counter", type = int, default = 0)
inputs = parser.parse_args()

counter = inputs.counter


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

# Fiducial parameters:

cosmo_params = p21c.CosmoParams({
        "SIGMA_8": 0.8102,       # ----------------------------------database---------------------------------------
        "hlittle": Planck18.h,
        "OMm": Planck18.Om0,
        "OMb": Planck18.Ob0,
        "POWER_INDEX": 0.9665,
})


user_params = p21c.UserParams({
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
})




flag_options = p21c.FlagOptions({
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
})



astro_params = p21c.AstroParams({
    "HII_EFF_FACTOR": 30.0,
    "F_STAR10": -1.3,   # ----------------------------------database---------------------------------------
    "F_STAR7_MINI": None,
    "ALPHA_STAR": 0.5,  # ----------------------------------database---------------------------------------
    "ALPHA_STAR_MINI": None,
    "F_ESC10": -1.0,     # ----------------------------------database---------------------------------------
    "F_ESC7_MINI": None, # This has default -2.0 but I am going to set it to None for the monent 
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
})


global_params={}


# # random_seed = 23
# redshift=5
# # redshift = 7.740863
# # redshift =12.513192
# z = redshift




# START------------------------------------------------------------------------------------------------------------------------------------
filename_fid = '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/save6/pm_fiducial_r23.h5'
savedir= '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/'
cache_location = '/leonardo_scratch/large/userexternal/ntriant1/database/_cache6/'



# # PS
# with h5py.File(filename_fid, 'r') as hdf:
#         lightcone_redshifts = hdf['lightcones'].attrs['lightcone_redshifts']
#         lc = hdf['lightcones']['brightness_temp'][:]
# ps=p21c_tools.calculate_ps(lc=lc,
#                     lc_redshifts = lightcone_redshifts,
#                     box_length = 300.0,
#                     box_side_shape = 200,
#                     calc_1d = True,
#                     calc_2d = False,
#                     )
# ps21_redshifts_fid = ps['redshifts']
# ps21_k01_fid = ps['ps_1D'][:, 5]
# ps21_k05_fid = ps['ps_1D'][:, 9]

# Properties
redshift=5
# redshift = 7.740863
# redshift =12.513192



# '''-----------------------------------Plus Minus-----------------------------------------------'''



# Here I am reading the dictionaries form 
import json
# Read the JSON file
with open('../parameter_dicts.txt', 'r') as file:
    parameter_dict = json.load(file)
# Accessing the dictionaries
parameter_dict_keys = list(parameter_dict.keys())

run_name=parameter_dict_keys[counter]
print(parameter_dict_keys, flush=True)

# # Determine which number based on what run actually run (stupid)
# # which_number='2'
# # Here fix for the things that are not in 2
# in1 = ['SIGMA_SFR_LIM_plus','F_STAR10_plus','ALPHA_STAR_plus','F_ESC10_plus','ALPHA_ESC_plus']
# in3 = ['SIGMA_8_plus','ALPHA_STAR_minus','M_TURN_plus' ]
# if (run_name in in1):
#         which_number=''
# elif (run_name in in3):
#         which_number='3'
# else:
#         which_number='2'
# print(f'{run_name} is in {which_number}', flush=True)

# Until here its stupid

# for run_name in run_names: 
which_number='6'
cache_location = f'/leonardo_scratch/large/userexternal/ntriant1/database/_cache{which_number}/_pm_{run_name}_cache/'
filename = f'/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/save{which_number}/pm_{run_name}_r23.h5'
if ('plus' in run_name):
        cmap = 'Blues'
        color = 'tab:blue'
        print(f'Determined color{color}', flush=True)
elif ('minus' in run_name):
        cmap = 'Reds'
        color = 'tab:red'
        print(f'Determined color{color}', flush=True)


# PS
with h5py.File(filename, 'r') as hdf:
        lightcone_redshifts = hdf['lightcones'].attrs['lightcone_redshifts']
        lc = hdf['lightcones']['brightness_temp'][:]
ps=p21c_tools.calculate_ps(lc=lc,
                lc_redshifts = lightcone_redshifts,
                box_length = 300.0,
                box_side_shape = 200,
                calc_1d = True,
                calc_2d = False,
                )
ps21_redshifts_fid = ps['redshifts']
ps21_k01_fid = ps['ps_1D'][:, 5]
ps21_k05_fid = ps['ps_1D'][:, 9]

np.save(f'/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/kps/{run_name}_01.npy', ps21_k01_fid)
np.save(f'/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/kps/{run_name}_05.npy', ps21_k05_fid)

if run_name=='fiducial':
       np.save(f'/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/kps/ps21_redshifts.npy', ps21_redshifts_fid)