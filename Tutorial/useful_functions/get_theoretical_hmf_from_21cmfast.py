#compare updating halo masses from previous redshifts to going straight from ICs
import numpy as np
import py21cmfast as p21c
from py21cmfast.c_21cmfast import lib

def get_theoretical_hmf(input_params_struct, redshift):

    inputs = input_params_struct
    lib.Broadcast_struct_global_all(
            inputs.user_params.cstruct,
            inputs.cosmo_params.cstruct,
            inputs.astro_params.cstruct,
            inputs.flag_options.cstruct,
        )

    edges = np.logspace(9,14,num=64)
    widths = np.diff(edges)
    dlnm = np.log(edges[1:]) - np.log(edges[:-1])
    centres = (edges[:-1] * np.exp(dlnm/2)).astype('f4')

    volume = inputs.user_params.BOX_LEN ** 3
    h_little = inputs.cosmo_params.cosmo.h
    rhocrit = inputs.cosmo_params.OMm * inputs.cosmo_params.cosmo.critical_density(0).to('Msun Mpc-3').value

    z = redshift
    lib.init_ps()
    growth_z = lib.dicke(z)

    lib.initialiseSigmaMInterpTable(edges[0]/2,edges[-1])
    umf = np.vectorize(lib.unconditional_mf)(
        growth_z,
        np.log(centres),
        z,
        inputs.user_params.cdict["HMF"]
    ) * rhocrit
    return(centres, umf)