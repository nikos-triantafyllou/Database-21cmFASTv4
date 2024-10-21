# This is a python script that plots the HMF from a specific h5 file as saved for the database, HMFs are already computed in the h5 files. 
# To see how they are computed check out the cooresponding file.

# PARAMETERS-------------------------------------------
filename = '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.59948425_r23.h5'
redshift_colorbar = True # If True in one plot all of the redshifts will be plotted 
# filenames= 0
# filename= 0
# filenames = ['smm_0.01_r23.h5',
#              'smm_0.027825594_r23.h5',
#               'smm_0.0166810054_r23.h5',
#               'smm_0.0774263683_r23.h5',
#               'smm_0.215443469_r23.h5',
#               'smm_0.359381366_r23.h5',
#               'smm_0.59948425_r23.h5',
#               'smm_0.0464158883_r23.h5',

# ]
filenames = ['/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.01_r23.h5',
             '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.0166810054_r23.h5',
            '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.0774263683_r23.h5',
             '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.215443469_r23.h5',
            #  '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.027825594_r23.h5',
             '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.359381366_r23.h5',
            #  '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.0464158883_r23.h5',
             '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.59948425_r23.h5'
]


savedir= '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/plots/HMFs/'


#-----------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
import h5py
from d_plotting import plot_hmf_z



if filename!=0 and redshift_colorbar:
    with h5py.File(filename, 'r') as hdf:
        node_redshifts = hdf['coeval_data'].attrs['node_redshifts']
        hist_mass_mean = hdf['coeval_data']['HMF_z'].attrs['hist_mass_mean']
        HMF_arr = hdf['coeval_data']['HMF_z'][:]
        smm= "%.2e" % dict(hdf['simulation_parameters']['user_params'].attrs)['SAMPLER_MIN_MASS']
    fig = plot_hmf_z(HMF_arr, hist_mass_mean, node_redshifts, smm)
    fig.savefig(f'{savedir}HMF_z/{smm}.pdf',bbox_inches='tight')
elif filenames!=0:
    for filename in filenames:
        with h5py.File(filename, 'r') as hdf:
            node_redshifts = hdf['coeval_data'].attrs['node_redshifts']
            hist_mass_mean = hdf['coeval_data']['HMF_z'].attrs['hist_mass_mean']
            HMF_arr = hdf['coeval_data']['HMF_z'][:]
            smm= "%.2e" % dict(hdf['simulation_parameters']['user_params'].attrs)['SAMPLER_MIN_MASS']

        plt.plot(hist_mass_mean, HMF_arr[91], alpha=1,label=smm)

        # Make the plot pretty
        plt.xlabel(r'$\rm M_{\rm halo}~[?]$')
        plt.ylabel(r'$\rm dn/dlnM~[?]$')
        plt.legend()
        # plt.title(label=smm)
        plt.xscale('log')
        plt.yscale('log')
    plt.savefig(f'{savedir}HMF_smms.pdf',bbox_inches='tight')