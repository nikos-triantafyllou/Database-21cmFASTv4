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
}

astro_params = p21c.AstroParams(astro_params_dict)


global_params={}


# # random_seed = 23
# redshift=5
# # redshift = 7.740863
# # redshift =12.513192
# z = redshift



# START------------------------------------------------------------------------------------------------------------------------------------
plt.rcParams.update({'font.size': 24}) 
plt.rcParams['lines.linewidth'] = 3.5


filename_fid = '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/save6/pm_fiducial_r23.h5'
savedir= '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/'
cache_location = '/leonardo_scratch/large/userexternal/ntriant1/database/_cache6/'

nrows=10
ncols=7
# Create a figure with a 10x11 grid of subplots

fig, ax = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(65, 65),
    gridspec_kw={'hspace': 0.0}  # More space between row 5 and 6
)

# fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(65, 30), sharex=False)  # Adjust figsize for better visibility
ynames= ['SIGMA8 ',
          r'$ \rm log_{10}(L_{X<2keV}/SFR / (erg \; s^{-1} \; M_\odot^{-1} \; yr) )$ ',
          r'NU_X_THRESH [eV]',
          r'$\rm log_{10}(F\_STAR10)$',
          'ALPHA_STAR',
          r'$\rm log_{10}(F\_ESC10)$',
          'ALPHA_ESC',
          't_STAR',
          'SIGMA_SFR_LIM',
          r'$\rm log_{10}(M\_TURN/M_\odot)$'
]

for i in range(nrows):
        ax[i,0].set_ylabel(ynames[i])

xnames = [r'$\rm x_{HI}$',
         r'$\rm T_{k, all\; gas} \; [mK]$',
          r'$\rm \delta T_b \; [mK]$',
          r'$\rm \Delta^2_{21}\ [mK^2]$' + "\n" + r'$\rm at\ k=0.1\ Mpc^{-1}$',
         r'$\rm \Delta^2_{21}\ [mK^2]$' + "\n" + r'$\rm at\ k=0.5\ Mpc^{-1}$',
          r'$\rm log_{10}(M_\star /M_\odot)$',
         r'$\rm log_{10}(SFR/(M_\odot \; s^{-1}))$',
         r'$\rm log_{10}(L_{X<2keV}/SFR / (erg \; s^{-1} \; M_\odot^{-1} \; yr) )$ ',
         r'$\rm log_{10} (M_\star/ M_\odot )$',
         r'$\rm log_{10}(SFR/ (M_\odot \; s^{-1}) $',
         r'$\rm log_{10}(L_{X<2keV}/SFR / (erg \; s^{-1} \; M_\odot^{-1} \; yr) )$ '
]


for i in range(ncols):
        ax[0,i].set_title(xnames[i])



# Add fiducial everywhere---------------------------------------------------------------
# Globals
with h5py.File(filename_fid, 'r') as hdf:
                node_redshifts_fid = hdf['coeval_data'].attrs['node_redshifts']
                Tb_global_fid = hdf['coeval_data']['Tb_z_global'][:]
                xH_global_fid = hdf['coeval_data']['EoR_history'][:]
                Tk_global_fid = hdf['coeval_data']['Tk_z_global'][:]
                Tk_all_gas_fid = hdf['coeval_data']['Tk_z_all_gas'][:]
                hmf           = hdf['coeval_data']['HMF_z'][-1,:]
                hmf_mean_mass  = hdf['coeval_data']['HMF_z'].attrs['hist_mass_mean']   
                hmf2            = hdf['coeval_data']['HMF_z'][-40,:]

# PS
with h5py.File(filename_fid, 'r') as hdf:
        lightcone_redshifts = hdf['lightcones'].attrs['lightcone_redshifts']
        lc = hdf['lightcones']['brightness_temp'][:]

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

ps21_redshifts_fid = np.load(f'/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/kps/ps21_redshifts.npy')


run_name='fiducial'
ps21_k01_fid = np.load(f'/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/kps/{run_name}_01.npy')
ps21_k05_fid = np.load(f'/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/kps/{run_name}_05.npy')


print(node_redshifts_fid )

# Properties
redshift=5
# redshift = 7.740863
# redshift =12.513192
from d_get_properties import get_properties
n_halos=100000
cache_location_fid = '/leonardo_scratch/large/userexternal/ntriant1/database/_cache6/_pm_fiducial_cache/'

import json
# Read the JSON file
with open('../parameter_dicts.txt', 'r') as file:
    parameter_dict = json.load(file)
varying_params = parameter_dict['fiducial']
cosmo_params.SIGMA_8 = varying_params['SIGMA_8']
for astro_key in list(varying_params.keys())[2:]:
        astro_params_dict[astro_key] = varying_params[astro_key]
astro_params = p21c.AstroParams(astro_params_dict)

# # # hm_cat, sm_cat, sfr_cat, metallicity, xray_cat = get_properties(redshift, n_halos, cache_location_fid, cosmo_params, user_params, flag_options, astro_params)

hm_cat, sm_cat, sfr_cat, metallicity, xray_cat = np.load(f'/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/properties/fiducial_props_5.npy')


# xx1, yy1, density1 = get_contour_stuff(hm_cat, sm_cat)
# xx2, yy2, density2 = get_contour_stuff(sm_cat, sfr_cat)
# xx3, yy3, density3 = get_contour_stuff(metallicity,xray_cat/sfr_cat*1e38)

xarr1, ymean1, yerr1 = get_mean_std_stuff(hm_cat, sm_cat, bins=100)
xarr2, ymean2, yerr2 = get_mean_std_stuff(sm_cat, sfr_cat, bins=100)
xarr3, ymean3, yerr3 = get_mean_std_stuff(metallicity,xray_cat/sfr_cat*1e38, bins=100)


hm_cat, sm_cat, sfr_cat, metallicity, xray_cat = np.load(f'/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/properties/fiducial_props_12.npy')

xarr1_z, ymean1_z, yerr1_z = get_mean_std_stuff(hm_cat, sm_cat, bins=100)
xarr2_z, ymean2_z, yerr2_z = get_mean_std_stuff(sm_cat, sfr_cat, bins=100)
xarr3_z, ymean3_z, yerr3_z = get_mean_std_stuff(metallicity,xray_cat/sfr_cat*1e38, bins=100)

# Loop through each subplot and plot data
for j in range(10): # columns (parameters, i.e. sims) # rows for portrait NOW J IS ROWS
    # filename_minus=
    # filename_plus=
    # for i in range(11): # rows summaries

    #Globals
    label=varying_params[list(varying_params.keys())[j+1]]
    ax[j, 0].plot(node_redshifts_fid, xH_global_fid, color = 'black', label=f'fiducial: {label}' )
    ax[j, 0].set_ylim(0, 1.1)
    ax[j, 0].axhline(1.0, color='black', linestyle= '--')
#     ax[0,j].set_xlabel('Redshift, z')

    ax[j, 2].plot(node_redshifts_fid, Tb_global_fid, color = 'black', label=f'fiducial: {label}' )
    ax[j, 2].set_ylim(-160, 25)
    ax[j, 2].axhline(0.0, label='y=0', color='black', linestyle= '--')
#     ax[1, j].legend()
    
#     ax[1,j].set_xlabel('Redshift, z')


    ax[j,1].plot(node_redshifts_fid, Tk_all_gas_fid, color = 'black' , label=f'fiducial: {label}')
#     ax[2,j].set_xlabel('Redshift, z')
    ax[j, 1].set_ylim(1, 1e5)
    ax[j,1].set_yscale('log')


    # Power Spectra
    ax[j,3].plot(ps21_redshifts_fid, ps21_k01_fid, color = 'black', label=f'fiducial: {label}')
#     ax[3,j].set_xlabel('Redshift, z')
    ax[j, 3].set_ylim(1e-3, 9e1)
    ax[j,3].set_yscale('log')
    

    ax[j,4].plot(ps21_redshifts_fid, ps21_k05_fid, color = 'black', label=f'fiducial: {label}' )
    ax[j,4].set_xlabel('Redshift, z')
    ax[j,4].set_ylim(1e-3, 9e1)
    ax[j,4].set_yscale('log')
    
#     ax[4,j].grid()


    # Contour stuff
#     ax[5,j].contourf(xx1, yy1, density1, levels=100, alpha=0.5, cmap='Greys', norm='log', nchunk=0, linestyles='none', antialiased=True)
#     percentile_90 = np.percentile(density1, 95)
#     ax[5,j].contourf(xx1, yy1, density1, levels=[percentile_90, np.max(density1)], alpha=0.3, colors='black', norm='log')
#     ax[5,j].set_xlabel(r'$\rm log_{10}(M_{halo}/M_\odot)$')
    
# #     ax[5,j].grid()


# #     ax[6,j].contourf(xx2, yy2, density2, levels=100, alpha=0.5, cmap='Greys', norm='log', nchunk=0, linestyles='none', antialiased=True)
#     percentile_90 = np.percentile(density2, 95)
#     ax[6,j].contourf(xx2, yy2, density2, levels=[percentile_90, np.max(density2)], alpha=0.3, colors='black', norm='log')
#     ax[6,j].set_xlabel(r'$\rm log_{10}(M_\star/ M_\odot)$')
# #     ax[6,j].grid()

# #     ax[7,j].contourf(xx3, yy3, density3, levels=100, alpha=0.5, cmap='Greys', norm='log', nchunk=0, linestyles='none', antialiased=True)
#     percentile_90 = np.percentile(density3, 95)
#     ax[7,j].contourf(xx3, yy3, density3, levels=[percentile_90, np.max(density3)], alpha=0.3, colors='black', norm='log')
#     ax[7,j].set_xlabel(r'$\rm log_{10}(Z/Zsun)$')
# #     ax[7,j].grid()
# #     Mean std stuff
    
    if j==0:
           ax[j,5].plot(hmf_mean_mass, hmf, color='black')
           ax[j,5].set_ylabel(r'$\rm dn/dlnM [Mpc^{-3}]$')
        #    ax[5,j].set_xlabel(r'$\rm M[M_\odot]$')
           ax[j,5].set_yscale('log')
           ax[j,5].set_xscale('log')

           ax[j,6].plot(hmf_mean_mass, hmf2,color='black')
           ax[j,6].set_ylabel(r'$\rm dn/dlnM [Mpc^{-3}]$')
           ax[j,6].set_xlabel(r'$\rm M[M_\odot]$')
           ax[j,6].set_yscale('log')
           ax[j,6].set_xscale('log')

    if j==1:
           ax[j,5].plot(xarr3, ymean3, c='black', label=label)
           ax[j,5].fill_between(xarr3, ymean3-1*yerr3, ymean3+1*yerr3, alpha=0.3, label=r'$1\sigma$',color= 'black')
        #    ax[5,j].set_xlabel(r'$\rm log_{10}(Z/Zsun)$')

           ax[j,6].plot(xarr3_z, ymean3_z, c='black', label= label)
           ax[j,6].fill_between(xarr3_z, ymean3_z-1*yerr3_z, ymean3_z+1*yerr3_z, alpha=0.3, label=r'$1\sigma$',color= 'black')
           ax[j,6].set_xlabel(r'$\rm log_{10}(Z/Zsun)$')

    if j==3 or j==4 or j==9:
           ax[j,5].plot(xarr1, ymean1, c='black', label= label)
           ax[j,5].fill_between(xarr1, ymean1-1*yerr1, ymean1+1*yerr1, alpha=0.3, label=r'$1\sigma$',color= 'black')
        #    ax[5,j].set_xlabel(r'$\rm log_{10}(M_{halo}/M_\odot)$')

           ax[j,6].plot(xarr1_z, ymean1_z, c='black', label= label)
           ax[j,6].fill_between(xarr1_z, ymean1_z-1*yerr1_z, ymean1_z+1*yerr1_z, alpha=0.3, label=r'$1\sigma$',color= 'black')
           ax[j,6].set_xlabel(r'$\rm log_{10}(M_{halo}/M_\odot)$')


    if j==7 or j==8:
           ax[j,5].plot(xarr2, ymean2, c='black', label=label)
           ax[j,5].fill_between(xarr2, ymean2-1*yerr2, ymean2+1*yerr2, alpha=0.3, label=r'$1\sigma$',color= 'black')
        #    ax[5,j].set_xlabel(r'$\rm log_{10}(M_\star / M_\odot)$')

           ax[j,6].plot(xarr2_z, ymean2_z, c='black', label=label)
           ax[j,6].fill_between(xarr2_z, ymean2_z-1*yerr2_z, ymean2_z+1*yerr2_z, alpha=0.3, label=r'$1\sigma$',color= 'black')
           ax[j,6].set_xlabel(r'$\rm log_{10}(M_\star / M_\odot)$')
    













































# '''-----------------------------------Plus Minus-----------------------------------------------'''



# Here I am reading the dictionaries form 
import json
# Read the JSON file
with open('../parameter_dicts.txt', 'r') as file:
    parameter_dict = json.load(file)
# Accessing the dictionaries
parameter_dict_keys = list(parameter_dict.keys())
run_names=parameter_dict_keys[1:]

j=0 # This determines the column
ii=0
for run_name in run_names:
        cosmo_params = p21c.CosmoParams({
        "SIGMA_8": 0.8102,       # ----------------------------------database---------------------------------------
        "hlittle": Planck18.h,
        "OMm": Planck18.Om0,
        "OMb": Planck18.Ob0,
        "POWER_INDEX": 0.9665,
        })

        astro_params_dict={
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
        }

        varying_params = parameter_dict[run_name]
        cosmo_params.SIGMA_8 = varying_params['SIGMA_8']
        for astro_key in list(varying_params.keys())[2:]:
                astro_params_dict[astro_key] = varying_params[astro_key]
        astro_params = p21c.AstroParams(astro_params_dict)
                
        which_number='6'
        cache_location = f'/leonardo_scratch/large/userexternal/ntriant1/database/_cache{which_number}/_pm_{run_name}_cache/'
        filename = f'/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/save{which_number}/pm_{run_name}_r23.h5'
        if ('plus' in run_name):
                cmap = 'Blues'
                color = 'teal'
                print(f'Determined color{color}', flush=True)
        elif ('minus' in run_name):
                cmap = 'Reds'
                color = 'darkgoldenrod'
                print(f'Determined color{color}', flush=True)



        # Globals
        label=varying_params[list(varying_params.keys())[j+1]]
        with h5py.File(filename, 'r') as hdf:
                node_redshifts = hdf['coeval_data'].attrs['node_redshifts']
                Tb_global = hdf['coeval_data']['Tb_z_global'][:]
                xH_global = hdf['coeval_data']['EoR_history'][:]
                Tk_global = hdf['coeval_data']['Tk_z_global'][:]
                Tk_all_gas = hdf['coeval_data']['Tk_z_all_gas'][:]
                hmf           = hdf['coeval_data']['HMF_z'][-1,:]
                hmf_mean_mass  = hdf['coeval_data']['HMF_z'].attrs['hist_mass_mean']   
                hmf2            = hdf['coeval_data']['HMF_z'][-40,:]
        
        ax[j,0].plot(node_redshifts, xH_global, color = color, label=label )
        # ax[0,j].set_xlabel('Redshift, z')
        
        ax[j,2].plot(node_redshifts, Tb_global, color = color, label=label )
        ax[j,2].set_ylim(-160, 25)
        # ax[1,j].set_xlabel('Redshift, z')
        
        ax[j,1].plot(node_redshifts, Tk_all_gas, color = color, label=label )
        # ax[2,j].set_xlabel('Redshift, z')


        ps21_k01_fid = np.load(f'/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/kps/{run_name}_01.npy')
        ps21_k05_fid = np.load(f'/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/kps/{run_name}_05.npy')

        # Power Spectra
        ax[j,3].plot(ps21_redshifts_fid, ps21_k01_fid, color = color, label=label )
        # ax[3,j].set_xlabel('Redshift, z')
        
        ax[j,4].plot(ps21_redshifts_fid, ps21_k05_fid, color = color , label=label)
        ax[j,4].set_xlabel('Redshift, z')


        
        # hm_cat, sm_cat, sfr_cat, metallicity, xray_cat = get_properties(redshift, n_halos, cache_location, cosmo_params, user_params, flag_options, astro_params)
        hm_cat, sm_cat, sfr_cat, metallicity, xray_cat = np.load(f'/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/properties/{run_name}_props_5.npy')


        # # Mean std stuff
        
        if j==0:
                ax[j,5].plot(hmf_mean_mass, hmf, color=color)
                # ax[5,j].set_xlabel(r'$\rm dn/dlnM [Mpc^{-3}]$')
                ax[j,5].set_ylabel(r'$\rm dn/dlnM [M_\odot]$')

        if j==1:
                xarr3, ymean3, yerr3 = get_mean_std_stuff(metallicity,xray_cat/sfr_cat*1e38, bins=100)
                ax[j,5].plot(xarr3, ymean3, c=color, label=label)
                ax[j,5].fill_between(xarr3, ymean3-1*yerr3, ymean3+1*yerr3, alpha=0.3, label=r'$1\sigma$',color= color)
                # ax[5,j].set_xlabel(r'$\rm log_{10}(Z/Zsun)$')

        if j==3 or j==4 or j==9:
                xarr1, ymean1, yerr1 = get_mean_std_stuff(hm_cat, sm_cat, bins=100)
                ax[j,5].plot(xarr1, ymean1,  c=color, label=label)
                ax[j,5].fill_between(xarr1, ymean1-1*yerr1, ymean1+1*yerr1, alpha=0.3, label=r'$1\sigma$',color= color)
                # ax[5,j].set_xlabel(r'$\rm log_{10}(M_{halo}/M_\odot)$')


        if j==7 or j==8:
                xarr2, ymean2, yerr2 = get_mean_std_stuff(sm_cat, sfr_cat, bins=100)
                ax[j,5].plot(xarr2, ymean2,  c=color, label=label)
                ax[j,5].fill_between(xarr2, ymean2-1*yerr2, ymean2+1*yerr2, alpha=0.3, label=r'$1\sigma$',color= color)
                # ax[5,j].set_xlabel(r'$\rm log_{10}(M_\star / M_\odot)$')


        hm_cat, sm_cat, sfr_cat, metallicity, xray_cat = np.load(f'/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_plus_minus/plots/properties/{run_name}_props_12.npy')


        if j==0:
                ax[j,6].plot(hmf_mean_mass, hmf2, color=color)
                ax[j,6].set_xlabel(r'$\rm dn/dlnM [Mpc^{-3}]$')
                ax[j,6].set_ylabel(r'$\rm dn/dlnM [M_\odot]$')

        if j==1:
                xarr3, ymean3, yerr3 = get_mean_std_stuff(metallicity,xray_cat/sfr_cat*1e38, bins=100)
                ax[j,6].plot(xarr3, ymean3, c=color, label=label)
                ax[j,6].fill_between(xarr3, ymean3-1*yerr3, ymean3+1*yerr3, alpha=0.3, label=r'$1\sigma$',color= color)
                ax[j,6].set_xlabel(r'$\rm log_{10}(Z/Zsun)$')

        if j==3 or j==4 or j==9:
                xarr1, ymean1, yerr1 = get_mean_std_stuff(hm_cat, sm_cat, bins=100)
                ax[j,6].plot(xarr1, ymean1,  c=color, label=label)
                ax[j,6].fill_between(xarr1, ymean1-1*yerr1, ymean1+1*yerr1, alpha=0.3, label=r'$1\sigma$',color= color)
                ax[j,6].set_xlabel(r'$\rm log_{10}(M_{halo}/M_\odot)$')


        if j==7 or j==8:
                xarr2, ymean2, yerr2 = get_mean_std_stuff(sm_cat, sfr_cat, bins=100)
                ax[j,6].plot(xarr2, ymean2,  c=color, label=label)
                ax[j,6].fill_between(xarr2, ymean2-1*yerr2, ymean2+1*yerr2, alpha=0.3, label=r'$1\sigma$',color= color)
                ax[j,6].set_xlabel(r'$\rm log_{10}(M_\star / M_\odot)$')


        if ii%2==1:
           j+=1     
        ii+=1        



# nrows=11
for i in range(ncols):
       for j in range(nrows):
              ax[j, i].tick_params(bottom=True, top=True, left=True, right=True)
              plt.tick_params(which='both', direction='inout', length=6, width=0.5, colors='black', grid_alpha=0.5)
              ax[j, i].tick_params(which='both', direction='inout', length=6, width=0.5, colors='black', grid_alpha=0.5)
              if i==0:
                ax[j,i].legend()
              if i<5:
                     ax[j, i].set_xlim(5, 35)
                     ax[j,i].set_xlabel('Redshift, z')
              if j!=9:
                     ax[j, i].set_xticklabels([])
# for i in [5,6]:
#     for j in range(ncols):
#         pos = ax[i, j].get_position()  # Get the position of the subplot
#         ax[i, j].set_position([pos.x0, pos.y0 - 0.06*(1), pos.width, pos.height])  # Shift down the last two rows



# Adjust the layout
# plt.tight_layout()

fig.savefig(f'{savedir}big_plus_minus_plot_small_portrait_7.pdf')