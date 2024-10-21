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

astro_params_dict={
    "HII_EFF_FACTOR": 30.0,
    "F_STAR10": -1.3,   # ----------------------------------database---------------------------------------
    "F_STAR7_MINI": None,
    "ALPHA_STAR": 0.5,  # ----------------------------------database---------------------------------------
    "ALPHA_STAR_MINI": None,
    "F_ESC10": -1.0,     # ----------------------------------database---------------------------------------
    "F_ESC7_MINI": -2.0, # This has default -2.0 but I am going to set it to None for the monent 
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

astro_params = p21c.AstroParams(astro_params_dict)


global_params={}


# # random_seed = 23
# redshift=5
# # redshift = 7.740863
# # redshift =12.513192
# z = redshift



# START------------------------------------------------------------------------------------------------------------------------------------
savedir= '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/'
filename_fid = '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/save5/pm_fiducial_r23.h5'
cache_location = '/leonardo_scratch/large/userexternal/ntriant1/database/_cache5/'

# Globals
with h5py.File(filename_fid, 'r') as hdf:
                node_redshifts_fid = hdf['coeval_data'].attrs['node_redshifts']
                log10_mturnovers = hdf['coeval_data'].attrs['log10_mturnovers']
                log10_mturnovers_mini = hdf['coeval_data'].attrs['log10_mturnovers_mini']
                LF_arr_0 = hdf['coeval_data']['UV_LF_0'][:]
                LF_arr_1= hdf['coeval_data']['UV_LF_1'][:]
                LF_arr_2 = hdf['coeval_data']['UV_LF_2'][:]


z_uv=node_redshifts_fid  
LF_arr=p21c.compute_luminosity_function(
    redshifts = z_uv,
    user_params=user_params,
    cosmo_params=cosmo_params,
    astro_params=astro_params,
    flag_options=flag_options,
    nbins=100,
    mturnovers=10**log10_mturnovers,
    mturnovers_mini=10**log10_mturnovers_mini,
    component=0,
)    
LF_arr_0 = LF_arr[0]
LF_arr_1 = LF_arr[1]
LF_arr_2 = LF_arr[2]


# Returns
#     -------
#     Muvfunc : np.ndarray
#         Magnitude array (i.e. brightness). Shape [nredshifts, nbins]
#     Mhfunc : np.ndarray
#         Halo mass array. Shape [nredshifts, nbins]
#     lfunc : np.ndarray
#         Number density of haloes corresponding to each bin defined by `Muvfunc`.
#         Shape [nredshifts, nbins].

print(node_redshifts_fid, flush=True)
# # UV_LF
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

# print(LF_arr)

# np.save('LF_arr.npy',LF_arr_0)



z_id=-1

print(LF_arr_2[z_id,:], flush=True)
print(LF_arr_0[z_id,:], flush=True)


fig, ax = plt.subplots()
ax.plot(LF_arr_0[z_id,:], 10**LF_arr_2[z_id,:]
        # /LF_arr_0[z_id,:]
        , color='black', label='from inputs')
ax.set_xlabel(r'$\rm M_{UV}$')
ax.set_ylabel(r'$\rm \phi /mag/Mpc^3$')
ax.set_title(     label = f'redshift={round(node_redshifts_fid[z_id], 1)}' )

ax.set_xlim((-25,-5))
ax.set_ylim((1e-5,1e2))
ax.set_yscale('log')
# ax.set_xscale('log')
# ax.grid()





LF_arr=p21c.compute_luminosity_function(
    redshifts = z_uv,
    user_params=user_params,
    cosmo_params=cosmo_params,
    astro_params=astro_params,
    flag_options=flag_options,
    nbins=100,
    mturnovers=10**log10_mturnovers,
    mturnovers_mini=10**log10_mturnovers_mini,
    component=1,
)    
LF_arr_0 = LF_arr[0]
LF_arr_1 = LF_arr[1]
LF_arr_2 = LF_arr[2]
ax.plot(LF_arr_0[z_id,:], 10**LF_arr_2[z_id,:]
        # /LF_arr_0[z_id,:]
        , color='tab:red', label='only massive')

LF_arr=p21c.compute_luminosity_function(
    redshifts = z_uv,
    user_params=user_params,
    cosmo_params=cosmo_params,
    astro_params=astro_params,
    flag_options=flag_options,
    nbins=100,
    mturnovers=10**log10_mturnovers,
    mturnovers_mini=10**log10_mturnovers_mini,
    component=2,
)    
LF_arr_0 = LF_arr[0]
LF_arr_1 = LF_arr[1]
LF_arr_2 = LF_arr[2]
ax.plot(LF_arr_0[z_id,:], 10**LF_arr_2[z_id,:]
        # /LF_arr_0[z_id,:]
        , color='tab:green', label='only mini')




#========================================From SFRs===================================================================
# Properties
redshift=5
# redshift = 7.740863
# redshift =12.513192
from d_get_properties import get_properties
n_halos=100000
cache_location_fid = '/leonardo_scratch/large/userexternal/ntriant1/database/_cache5/_pm_fiducial_cache/'

import json
# Read the JSON file
with open('../parameter_dicts.txt', 'r') as file:
    parameter_dict = json.load(file)
varying_params = parameter_dict['fiducial']
cosmo_params.SIGMA_8 = varying_params['SIGMA_8']
for astro_key in list(varying_params.keys())[2:]:
        astro_params_dict[astro_key] = varying_params[astro_key]
astro_params = p21c.AstroParams(astro_params_dict)

hm_cat, sm_cat, sfr_cat, metallicity, xray_cat = get_properties(redshift, n_halos, cache_location_fid, cosmo_params, user_params, flag_options, astro_params)

# hm_cat, sm_cat, sfr_cat, metallicity, xray_cat = np.load(f'/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/properties/fiducial_props_5.npy')

Luv_over_SFR = 1.0 / 1.15 / 1e-28

Muv_cat = 51.63 - 2.5*np.log10(sfr_cat*Luv_over_SFR)


hist, bin_edges = np.histogram(Muv_cat, bins=100)
print(hist,flush=True)
# Compute the bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[3]-bin_edges[2]

# Normalize the histogram by the survey volume and bin width (for /mag/Mpc^3)
sim_volume= 300*300*300
hist = hist/ (bin_width * sim_volume)


# Normalize the histogram by the number of sampled halos out of the total number 
# All the halos are: 264636275
hist*= 264636275/100000



# fig2, ax2 = plt.subplots()

# Plot the luminosity function (log-log plot)
# plt.figure(figsize=(8, 6))
ax.plot(bin_centers, hist, drawstyle='steps-mid',color='teal', label='from halos')
# ax2.set_yscale('log')

ax.legend()
# fig2.savefig('histuv.png')
fig.savefig('uvlf.png')