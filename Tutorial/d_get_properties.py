import argparse
import numpy as np

import py21cmfast as p21c
from py21cmfast.c_21cmfast import ffi, lib

import logging, sys, os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from matplotlib import pyplot as plt
from timeit import default_timer as timer
from py21cmfast.cache_tools import readbox


# import d_utils


def get_properties(redshift, random_seed, n_halos, hdf_group, cosmo_params, user_params, flag_options, astro_params):
    z=redshift
    
#     pt_halos_ = perturbed_halo_field
    #fake pt_halos
    # n_halos = 1000000
    pt_halos = p21c.PerturbHaloField(
                    redshift= redshift,
                    random_seed = random_seed,
                    user_params = user_params,
                    cosmo_params = cosmo_params,
                    astro_params = astro_params,
                    flag_options = flag_options,
                    buffer_size = n_halos
            )
    setattr(pt_halos,"n_halos",n_halos)

    pt_halos()
    random_indices = np.random.choice(len(hdf_group['halo_masses'][:]), size=n_halos, replace=False)
    # Use the random indices to select elements
    pt_halos.halo_masses[...] = hdf_group['halo_masses'][:][random_indices]
    pt_halos.star_rng[...] = hdf_group['star_rng'][:][random_indices]
    pt_halos.sfr_rng[...] = hdf_group['sfr_rng'][:][random_indices]
#     pt_halos.xray_rng[...] = pt_halos_.xray_rng[random_indices]
    pt_halos()


    zero_array = ffi.cast("float *", np.zeros(1).ctypes.data)
    props_out = np.zeros(int(1e8)).astype('f4')
    lib.test_halo_props(
            float(z),
            user_params.cstruct,
            cosmo_params.cstruct,
            astro_params.cstruct,
            flag_options.cstruct,
            zero_array,
            zero_array,
            zero_array,
            zero_array,
            pt_halos(),
            ffi.cast("float *", props_out.ctypes.data),
    )
    
    props_out = props_out[:pt_halos.n_halos*12]
    props_out = props_out.reshape((pt_halos.n_halos,12))

    s_per_yr = 86400 * 365.25
    hm_cat = props_out[:,0]
    sm_cat = props_out[:,1]
    sfr_cat = props_out[:,2]*s_per_yr

    xray_cat = props_out[:,3].astype('f8')
    nion_cat = props_out[:,4]
    wsfr_cat = props_out[:,5]
    sm_mini = props_out[:,6]
    sfr_mini = props_out[:,7]
    mturn_a = props_out[:,8]
    mturn_m = props_out[:,9]
    mturn_r = props_out[:,10]
    metallicity = props_out[:,11]

    sel = np.any(~np.isfinite(props_out),axis=-1)
    if sel.sum() > 0:
            print(f'{sel.sum()} invalid halos')
            print(f'First 10: {props_out[sel,:][:10,:]}')
    return hm_cat, sm_cat, sfr_cat, 
# metallicity, xray_cat



































# Garbage------------------------------------------------------------------------------------------------------------------------------------

def get_properties_from_cache(redshift, n_halos, cache_location, cosmo_params, user_params, flag_options, astro_params):
    z=redshift
    
    names = d_utils.search_files(folder= cache_location, pattern='PerturbHalo*')
    # print(names, flush=True)
    for n in names:
        perturbed_halo_field=readbox(direc = cache_location,
            fname = n)
        # load_data = False)
    #     print(halo_field.redshift)
        if (abs(perturbed_halo_field.redshift-redshift)<10e-4):
            print('found it')
            break
    pt_halos_ = perturbed_halo_field
    #fake pt_halos
    # n_halos = 1000000
    pt_halos = p21c.PerturbHaloField(
                    redshift= pt_halos_.redshift,
                    user_params = user_params,
                    cosmo_params = cosmo_params,
                    astro_params = astro_params,
                    flag_options = flag_options,
                    buffer_size= n_halos,
                    random_seed = pt_halos_.random_seed
            )
    setattr(pt_halos,"n_halos",n_halos)

    # pt_halos()
    # pt_halos.halo_masses[...] = pt_halos_.halo_masses[:n_halos]
    # pt_halos.star_rng[...] = pt_halos_.star_rng[:n_halos]
    # pt_halos.sfr_rng[...] = pt_halos_.sfr_rng[:n_halos]
    # pt_halos.xray_rng[...] = pt_halos_.xray_rng[:n_halos]
    # pt_halos()
    print('All the halos are:', len(pt_halos_.halo_masses), flush=True)

    pt_halos()
    random_indices = np.random.choice(len(pt_halos_.halo_masses), size=n_halos, replace=False)
    # Use the random indices to select elements
    pt_halos.halo_masses[...] = pt_halos_.halo_masses[random_indices]
    pt_halos.star_rng[...] = pt_halos_.star_rng[random_indices]
    pt_halos.sfr_rng[...] = pt_halos_.sfr_rng[random_indices]
    pt_halos.xray_rng[...] = pt_halos_.xray_rng[random_indices]
    pt_halos()


    zero_array = ffi.cast("float *", np.zeros(1).ctypes.data)
    props_out = np.zeros(int(1e10)).astype('f4')
    lib.test_halo_props(
            z,
            user_params(),
            cosmo_params(),
            astro_params(),
            flag_options(),
            zero_array,
            zero_array,
            zero_array,
            zero_array,
            pt_halos(),
            ffi.cast("float *", props_out.ctypes.data),
    )
    
    props_out = props_out[:pt_halos.n_halos*12]
    props_out = props_out.reshape((pt_halos.n_halos,12))

    s_per_yr = 86400 * 365.25
    hm_cat = props_out[:,0]
    sm_cat = props_out[:,1]
    sfr_cat = props_out[:,2]*s_per_yr

    xray_cat = props_out[:,3].astype('f8')
    nion_cat = props_out[:,4]
    wsfr_cat = props_out[:,5]
    sm_mini = props_out[:,6]
    sfr_mini = props_out[:,7]
    mturn_a = props_out[:,8]
    mturn_m = props_out[:,9]
    mturn_r = props_out[:,10]
    metallicity = props_out[:,11]

    sel = np.any(~np.isfinite(props_out),axis=-1)
    if sel.sum() > 0:
            print(f'{sel.sum()} invalid halos')
            print(f'First 10: {props_out[sel,:][:10,:]}')
    return hm_cat, sm_cat, sfr_cat, metallicity, xray_cat

     