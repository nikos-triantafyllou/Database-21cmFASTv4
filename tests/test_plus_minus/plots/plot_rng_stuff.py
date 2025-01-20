#compare updating halo masses from previous redshifts to going straight from ICs
import argparse
import numpy as np

import py21cmfast as p21c
from py21cmfast.c_21cmfast import ffi, lib

import logging, sys, os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from matplotlib import pyplot as plt
from timeit import default_timer as timer
from datetime import timedelta

# parser = argparse.ArgumentParser(description='test halo time correlations')
# parser.add_argument('--hires', type=int, help='DIM in py21cmfast, length of hires grid')
# parser.add_argument('--lores', type=int, help='HII_DIM in py21cmfast, length of lores grid')
# parser.add_argument('--boxlen', type=int, help='BOX_LEN in py21cmfast, size of box')

# parser.add_argument('--redshift', type=float, help='start redshift',default=6)
# parser.add_argument('--nthreads',type=int, default=1, help='number of OMP threads')

# args = parser.parse_args()

# if not os.path.exists('_cachej'):
#     os.mkdir('_cachej')

# p21c.config['direc'] = '_cachej'

# redshift=5
# #setup inputs and parameters
# cosmo_params = p21c.CosmoParams()
# user_params = p21c.UserParams(USE_INTERPOLATION_TABLES=True,N_THREADS=1,BOX_LEN=300.0,DIM=None,HII_DIM=200,HMF=0,STOC_MINIMUM_Z=-1)
# flag_options = p21c.FlagOptions(USE_HALO_FIELD=True,HALO_STOCHASTICITY=True,USE_MASS_DEPENDENT_ZETA=True,USE_TS_FLUCT=True)

# # astro_params_fixed = p21c.AstroParams(SIGMA_STAR=0.,SIGMA_SFR_LIM=0.,SIGMA_SFR_INDEX=0.,SIGMA_LX=0.,L_X=40.)
# astro_params_0 = p21c.AstroParams(SIGMA_STAR=0.,SIGMA_SFR_LIM=0.,SIGMA_SFR_INDEX=0.,SIGMA_LX=0.)
# astro_params_1 = p21c.AstroParams(SIGMA_SFR_LIM=0.,SIGMA_SFR_INDEX=0.,SIGMA_LX=0.)
# astro_params_2 = p21c.AstroParams(SIGMA_LX=0.)
# astro_params_final = p21c.AstroParams()
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

# _a_B_c2 = (4 * const.sigma_sb / const.c**3).cgs.value
# def Tcmb0(Ogamma0):
#         """Tcmb; the temperature of the CMB at z=0."""
#         return pow ( Ogamma0 *  Planck18.critical_density0.value / _a_B_c2 , 1/4)

# Planck18 = Planck15.clone(
#     Om0=(0.02242 + 0.11933) / 0.6766**2,
#     Ob0=0.02242 / 0.6766**2,
#     H0=67.66,
#     name="Planck18",
#     Neff = 0,
#     Tcmb0=Tcmb0(8.600000001024455e-05)
# )

import d_utils
from py21cmfast.cache_tools import readbox


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
})


global_params={}


# random_seed = 23
redshift=5
# redshift = 7.740863
# redshift =12.513192
z = redshift


# # random_seed = np.random.randint(1000)



# names = d_utils.search_files(folder= '../_cache/', pattern='HaloField*')
# print(names, flush=True)
# for n in names:
#     halo_field=readbox(direc = '../_cache',
#         fname = n,
#        load_data = False)
# #     print(halo_field.redshift)
#     if (halo_field.redshift-5<10e-8):
#         print('found it')
#         break



# names = d_utils.search_files(folder= '../_cache/', pattern='IonizedBox*')
# print(names, flush=True)
# for n in names:
#     ionized_box=readbox(direc ='../_cache/',
#         fname = n,
#        load_data = False)
# #     print(halo_field.redshift)
#     if (ionized_box.redshift-5<10e-8):
#         print('found it')
#         break




# names = d_utils.search_files(folder= '../_cache/', pattern='TsBox*')
# print(names, flush=True)
# for n in names:
#     ts_box=readbox(direc = '../_cache/',
#         fname = n,
#        load_data = False)
# #     print(halo_field.redshift)
#     if (ts_box.redshift-5<10e-8):
#         print('found it')
#         break

cache_location = '/leonardo_scratch/large/userexternal/ntriant1/database/_cache5/_pm_fiducial_cache/'
names = d_utils.search_files(folder= cache_location, pattern='PerturbHalo*')
print(names, flush=True)
for n in names:
    perturbed_halo_field=readbox(direc = cache_location,
        fname = n,
       load_data = False)
#     print(halo_field.redshift)
    if (abs(perturbed_halo_field.redshift-redshift)<10e-4):
        print('found it')
        break


print(perturbed_halo_field.redshift,flush=True)

pt_halos = perturbed_halo_field

pt_halos_ = perturbed_halo_field


'''comented out'''

# start = timer()
# ct = 0

# #fake pt_halos


# n_halos = 1000000

# pt_halos = p21c.PerturbHaloField(
#                 redshift= pt_halos_.redshift,
#                 user_params = user_params,
#                 cosmo_params = cosmo_params,
#                 astro_params = astro_params,
#                 flag_options = flag_options,
#                 buffer_size= n_halos,
#                 random_seed = pt_halos_.random_seed
#         )
# setattr(pt_halos,"n_halos",n_halos)
# # pt_halos()
# # pt_halos.halo_masses[...] = pt_halos_.halo_masses[:n_halos]
# # pt_halos.star_rng[...] = pt_halos_.star_rng[:n_halos]
# # pt_halos.sfr_rng[...] = pt_halos_.sfr_rng[:n_halos]
# # pt_halos.xray_rng[...] = pt_halos_.xray_rng[:n_halos]
# # pt_halos()

# pt_halos()
# random_indices = np.random.choice(len(pt_halos_.halo_masses), size=n_halos, replace=False)
# # Use the random indices to select elements
# pt_halos.halo_masses[...] = pt_halos_.halo_masses[random_indices]
# pt_halos.star_rng[...] = pt_halos_.star_rng[random_indices]
# pt_halos.sfr_rng[...] = pt_halos_.sfr_rng[random_indices]
# pt_halos.xray_rng[...] = pt_halos_.xray_rng[random_indices]
# pt_halos()

# # setattr(pt_halos,"n_halos",n_halos)
# # pt_halos()
# # pt_halos.halo_coords=pt_halos.halo_coords[:n_halos]
# # pt_halos.halo_masses = pt_halos.halo_masses[:n_halos]
# # pt_halos.star_rng = pt_halos.star_rng[:n_halos]
# # pt_halos.sfr_rng = pt_halos.sfr_rng[:n_halos]
# # pt_halos.xray_rng = pt_halos.xray_rng[:n_halos]
# # pt_halos.buffer_size = n_halos
# # pt_halos()

# test_paramater_sets = (astro_params, astro_params)

# #CAT PLOTS
# fig,axs = plt.subplots(len(test_paramater_sets),6,figsize=(16,12*len(test_paramater_sets)/6.), layout='constrained')
# fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, hspace=0.0,
#                             wspace=0.0)

# for i,row in enumerate(axs):
#     row[0].set_ylabel('stars')
#     row[1].set_ylabel('sfr')
#     row[2].set_ylabel('LX')
#     row[3].set_ylabel('nion')
#     row[4].set_ylabel('Z/Zsun')
#     row[5].set_ylabel('Lx/SFR')
      
#     row[0].set_xlabel('mass')
#     row[1].set_xlabel('stars')
#     row[2].set_xlabel('sfr')
#     row[3].set_xlabel('stars')
#     row[4].set_xlabel('stars')
#     row[5].set_xlabel('Z/Zsun')
#     for j,ax in enumerate(row):
#         ax.set_yscale('log')
#         ax.set_xscale('log')
#         ax.grid()

# zero_array = ffi.cast("float *", np.zeros(1).ctypes.data)
# for ap in test_paramater_sets:
#         props_out = np.zeros(int(1e10)).astype('f4')
#         lib.test_halo_props(
#                 z,
#                 user_params(),
#                 cosmo_params(),
#                 ap(),
#                 flag_options(),
#                 zero_array,
#                 zero_array,
#                 zero_array,
#                 zero_array,
#                 pt_halos(),
#                 ffi.cast("float *", props_out.ctypes.data),
#         )
        
#         props_out = props_out[:pt_halos.n_halos*12]
#         props_out = props_out.reshape((pt_halos.n_halos,12))

#         s_per_yr = 86400 * 365.25
#         hm_cat = props_out[:,0]
#         sm_cat = props_out[:,1]
#         sfr_cat = props_out[:,2]*s_per_yr

#         xray_cat = props_out[:,3].astype('f8')
#         nion_cat = props_out[:,4]
#         wsfr_cat = props_out[:,5]
#         sm_mini = props_out[:,6]
#         sfr_mini = props_out[:,7]
#         mturn_a = props_out[:,8]
#         mturn_m = props_out[:,9]
#         mturn_r = props_out[:,10]
#         metallicity = props_out[:,11]

#         sel = np.any(~np.isfinite(props_out),axis=-1)
#         if sel.sum() > 0:
#                 print(f'{sel.sum()} invalid halos')
#                 print(f'First 10: {props_out[sel,:][:10,:]}')

#         end = timer()
#         print(f'done cat {ct} | {timedelta(seconds=end-start)}',flush=True)
        
#         axs[ct,0].scatter(hm_cat,sm_cat,s=2)
#         axs[ct,0].set_title(rf'$\mu = ${np.log10(sm_cat).mean():.2e} $\sigma = ${np.log10(sm_cat).std():.2e}')
        
#         axs[ct,1].scatter(sm_cat,sfr_cat,s=2)
#         axs[ct,1].set_title(rf'$\mu = ${np.log10(sfr_cat).mean():.2e} $\sigma = ${np.log10(sfr_cat).std():.2e}')

#         axs[ct,2].scatter(sfr_cat,xray_cat*1e38,s=2)
#         lx_base = np.linspace(sfr_cat.min(),sfr_cat.max(),num=20)
#         lx_fit = lx_base*(10**ap.L_X)
#         axs[ct,2].plot(lx_base,lx_fit,'r-')
#         axs[ct,2].set_title(rf'$\mu = ${np.log10(xray_cat*1e38).mean():.2e} $\sigma = ${np.log10(xray_cat*1e38).std():.2e}')
        
#         axs[ct,3].scatter(sm_cat,nion_cat,s=2)
#         axs[ct,3].set_title(rf'$\mu = ${np.log10(metallicity).mean():.2e} $\sigma = ${np.log10(metallicity).std():.2e}')

#         axs[ct,4].scatter(sm_cat,metallicity,s=2)
#         axs[ct,4].set_title(rf'$\mu = ${np.log10(nion_cat).mean():.2e} $\sigma = ${np.log10(nion_cat).std():.2e}')

#         z_base = np.linspace(metallicity.min(),metallicity.max(),num=20)
#         axs[ct,5].scatter(metallicity,xray_cat/sfr_cat*1e38,s=2)
#         axs[ct,5].plot(z_base,lx_fit/lx_base,'r-')
#         axs[ct,5].set_title(rf'$\mu = ${np.log10(xray_cat/sfr_cat*1e38).mean():.2e} $\sigma = ${np.log10(xray_cat/sfr_cat*1e38).std():.2e}')

#         print(xray_cat[:10])
#         print(sfr_cat[:10])
#         print((xray_cat/sfr_cat*1e38)[:10])

#         ct += 1

# fig.savefig('./cat_scatter.png')


# # fig, ax = plt.subplots()
# # ax.scatter(sm_cat,sfr_cat,color='black')
# # # plt.title(label=r'SFMS')
# # plt.xlabel(r'$\rm M_{star}$')
# # plt.ylabel(r'$\rm SFR$')
# # plt.yscale('log')
# # plt.xscale('log')
# # plt.grid()


# # Mstar vs Mhalo----------------------------------------------------------------------------------------------------------
# fig, ax = plt.subplots()
# ax.scatter(hm_cat,sm_cat,color='black')
# # plt.title(label=r'$\rm M_{star} \; vs \; M_{halo}$')
# plt.xlabel(r'$\rm M_{halo}$')
# plt.ylabel(r'$\rm M_\star$')
# plt.yscale('log')
# plt.xscale('log')
# plt.grid()
# fig.savefig('./mstar_mhalo.png')

# fig, ax = plt.subplots()
# plt.hist2d(np.log10(hm_cat), np.log10(sm_cat),bins=70,alpha=1,
#                      cmap='viridis',
#                       norm='log'
#                     )
# plt.xlabel(r'$\rm log(M_{halo})$')
# plt.ylabel(r'$\rm log(M_\star)$')
# plt.colorbar(label=f'Number of points out of {n_halos}', aspect=25)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.grid()
# fig.savefig('./mstar_mhalo_hist.png')

# # SFR vs Mstar----------------------------------------------------------------------------------------------------------
# fig, ax = plt.subplots()
# ax.scatter(sm_cat,sfr_cat,color='black')
# # plt.title(label=r'$\rm M_{star} \; vs \; M_{halo}$')
# plt.xlabel(r'$\rm M_\star$')
# plt.ylabel(r'$\rm SFR$')
# plt.yscale('log')
# plt.xscale('log')
# plt.grid()
# fig.savefig('./sfr_mstar.png')

# fig, ax = plt.subplots()
# plt.hist2d(np.log10(sm_cat), np.log10(sfr_cat),bins=70,alpha=1,
#                      cmap='viridis',
#                       norm='log'
#                     )
# plt.xlabel(r'$\rm log(M_\star)$')
# plt.ylabel(r'$\rm log(SFR)$')
# plt.colorbar(label=f'Number of points out of {n_halos}', aspect=25)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.grid()
# fig.savefig('./sfr_mstar_hist.png')



# # Lx/SFR vs Z/Zsun----------------------------------------------------------------------------------------------------------
# fig, ax = plt.subplots()
# ax.scatter(metallicity,xray_cat/sfr_cat*1e38,color='black')
# # plt.title(label=r'$\rm M_{star} \; vs \; M_{halo}$')
# ax.plot(z_base,lx_fit/lx_base,'r-')
# plt.xlabel(r'$\rm L_X/SFR$')
# plt.ylabel(r'$\rm Z/Zsun$')
# plt.yscale('log')
# plt.xscale('log')
# plt.grid()
# fig.savefig('./LXSFR_ZZsun.png')

# fig, ax = plt.subplots()
# plt.hist2d(np.log10(metallicity),np.log10(xray_cat/sfr_cat*1e38),bins=70,alpha=1,
#                      cmap='viridis',
#                       norm='log'
#                     )
# plt.plot(np.log10(z_base),np.log10(lx_fit/lx_base),'r-')
# plt.ylabel(r'$\rm log(L_X/SFR)$')
# plt.xlabel(r'$\rm log(Z/Zsun)$')
# plt.colorbar(label=f'Number of points out of {n_halos}', aspect=25)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.grid()
# fig.savefig('./LXSFR_ZZsun_hist.png')



'''comented out'''











#HMF


BoxSize = 300.0 # Mpc/h
min_mass = 1e8 #minimum mass in Msun/h
max_mass = 1e14 #maximum mass in Msun/h
bins     = 40   #number of bins in the HMF
bins_mass = np.logspace(np.log10(min_mass), np.log10(max_mass), bins+1)
mass_mean = 10**(0.5*(np.log10(bins_mass[1:])+np.log10(bins_mass[:-1])))
dM        = bins_mass[1:] - bins_mass[:-1]


hmf = mass_mean * np.histogram(pt_halos_.halo_masses, bins=bins_mass)[0]/(dM*BoxSize**3)
    






from astropy import units as u
# rho_crit = Planck18.critical_density(redshift).to(u.Mpc**(-3) * u.Msun).value

# HMF---------------------------------------------------------------------------------------------------------
# Fix Planck cosmology for 21cmFAST
Planck18 = Planck15.clone(
    Om0=(0.02242 + 0.11933) / 0.6766**2,
    Ob0=0.02242 / 0.6766**2,
    H0=67.66,
    name="Planck18",
)

# _a_B_c2 = (4 * const.sigma_sb / const.c**3).cgs.value
# def Tcmb0(Ogamma0):
#         """Tcmb; the temperature of the CMB at z=0."""
#         return pow ( Ogamma0 *  Planck18.critical_density0.value / _a_B_c2 , 1/4)

# Planck18 = Planck15.clone(
#     Om0=(0.02242 + 0.11933) / 0.6766**2,
#     Ob0=0.02242 / 0.6766**2,
#     H0=67.66,
#     name="Planck18",
#     Neff = 0,
#     Tcmb0=Tcmb0(8.600000001024455e-05)
# )
import hmf as hmfs
lnkmin, lnkmax, dlnk = np.log(1e-8), np.log(1e5), 0.01
c2 = 931.46 * 10e6 # eV
mf = hmfs.MassFunction(z               = redshift,
                  transfer_model  = 'EH_NoBAO', 
                  lnk_min         = lnkmin, 
                  lnk_max         = lnkmax, 
                  dlnk            = dlnk, 
#                   transfer_params = {"use_sugiyama_baryons": True}, 
                  cosmo_model     = Planck18,
#                   cosmo_params    = {'Om0': OMm-OMnu, 
#                                      'Ob0' : OMb, 
# #                                      'Ol0' : 1. - OMm,
#                                      'H0' : H_0,
# #                                      'Ode0': OMl,
#                                      'Tcmb0': THETA27*2.7,
#                                      'Neff': Nnu,
#                                      'm_nu': 91.5 * OMnu* pow(hlittle, 2)* pow(Nnu, -1) * pow(c2, -1)   , # eV
#                                      'Ob0': OMb  
#                                       },
#                   n               = PS_index_n,
#                   sigma_8         = SIGMA8,
                  takahashi       = False,
                  growth_model    = hmfs.cosmology.growth_factor.Carroll1992,
                  growth_params   = {},
                  hmf_model       = hmfs.mass_function.fitting_functions.PS,
                  hmf_params      = {},
                  filter_model    = hmfs.density_field.filters.TopHat,
                  filter_params   = {},
                  disable_mass_conversion = False,
#                   mdef_model      = hmf.halos.mass_definitions.SOCritical,
                  mdef_params     = {},
                  Mmin            = 8,
                  Mmax            = 13,
                  dlog10m         = 0.01,
#                   delta_c         = delta_c(REDSHIFT)
                  
                
                  )


mf = hmfs.MassFunction(z           = redshift,
                  transfer_model  = 'BBKS', 
                  lnk_min         = lnkmin, 
                  lnk_max         = lnkmax, 
                  dlnk            = dlnk, 
                  transfer_params = {"use_sugiyama_baryons": True}, 
                  cosmo_model     = Planck18,
#                   cosmo_params    = {'Om0': OMm-OMnu, 
#                                      'Ob0' : OMb, 
# #                                      'Ol0' : 1. - OMm,
#                                      'H0' : H_0,
# #                                      'Ode0': OMl,
#                                      'Tcmb0': THETA27*2.7,
#                                      'Neff': Nnu,
#                                      'm_nu': 91.5 * OMnu* pow(hlittle, 2)* pow(Nnu, -1) * pow(c2, -1)   , # eV
#                                      'Ob0': OMb  
#                                       },
#                   n               = PS_index_n,
#                   sigma_8         = SIGMA8,
                  takahashi       = False,
                  growth_model    = hmfs.cosmology.growth_factor.Carroll1992,
                  growth_params   = {},
                  hmf_model       = hmfs.mass_function.fitting_functions.ST,
                  hmf_params      = {'A' : 0.353,
              'p' : 0.175,
              'a' : 0.707},
                  filter_model    = hmfs.density_field.filters.TopHat,
                  filter_params   = {},
                  disable_mass_conversion = False,
#                   mdef_model      = hmf.halos.mass_definitions.SOCritical,
                  mdef_params     = {},
                  Mmin            = 8,
                  Mmax            = 13,
                  dlog10m         = 0.01,
#                   delta_c         = delta_c(1)
                  
                
                  )
print(mf.cosmo)

BOX_LEN = user_params.BOX_LEN
HII_DIM = user_params.HII_DIM
OMm = cosmo_params.OMm

rho_crit = Planck18.critical_density(0).to(u.Mpc**(-3) * u.Msun).value
cell_mass = rho_crit * Planck18.Om(0) * (BOX_LEN / HII_DIM)**3

fig, ax = plt.subplots()
# hmf = hmf/ Planck18.h**3
ax.plot(mass_mean, hmf, color='teal', label='hmf')
ax.plot(mf.m/ Planck18.h, mf.dndlnm* Planck18.h**3 , label= 'Murray+2013', linestyle='--', color='tab:red')
ax.axvline(cell_mass, label='cell mass', color='black', linestyle= '--')
# plt.title(label=r'$\rm M_{star} \; vs \; M_{halo}$')
plt.xlabel(r'$\rm M_{\rm halo} \; [M_\odot]$')
plt.ylabel(r'$\rm dn/dlnM [Mpc^{-3}]$')
plt.legend()
# plt.title(label=smm)
plt.xscale('log')
plt.yscale('log')






























fig.savefig('./hmf.png')