

# PARAMETERS-------------------------------------------
# filename = '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.01_r23.h5'
redshift_colorbar = True # If True in one plot all of the redshifts will be plotted 

filenames = ['/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.01_r23.h5',
             '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.0166810054_r23.h5',
            '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.0774263683_r23.h5',
             '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.215443469_r23.h5',
            #  '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.027825594_r23.h5',
             '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.359381366_r23.h5',
            #  '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.0464158883_r23.h5',
             '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.59948425_r23.h5'
]

savedir= '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/plots/LCs/'



#-----------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
import h5py
import d_plotting
import numpy as np

i=0
for filename in filenames:
    with h5py.File(filename, 'r') as hdf:
            lightcone_redshifts = hdf['lightcones'].attrs['lightcone_redshifts']
            lc = hdf['lightcones']['brightness_temp'][:]
            smm= "%.2e" % dict(hdf['simulation_parameters']['user_params'].attrs)['SAMPLER_MIN_MASS']
    fig = d_plotting.plot_lc(lc,  lightcone_redshifts, smm)
    fig.savefig(f'{savedir}LC_smm_{smm}.pdf', bbox_inches='tight')
    if i==0:
        lc1=lc
        smm1=smm
        i+=1


lightcone = lc
redshift_arr = lightcone_redshifts

fig, ax = plt.subplots(figsize=((10,8)))
# plt.imshow(lc1[:,123,:]-lc[:,123,:],cmap='EoR',vmin = -150,vmax=30, extent=[0,2553,0,300])
plt.imshow(lc1[:,123,:]-lc[:,123,:], cmap='RdBu_r', 
        #    norm='symlog'
           )
ax.set_aspect(1.5)

zs= np.concatenate((np.arange(5,15,1), np.arange(15,23,2)))
zs = np.concatenate((zs,np.arange(23,37,3)))
zs=[5,6,7,8,9,11,13,16,20,26,35]

z_positions=[]
for z in zs:
    z_positions.append( np.where(abs(redshift_arr-z) ==  min(abs(redshift_arr-z)))[0][0] )

ax.set_xticks(ticks = z_positions, labels= zs )
# ax.set_yticks(ticks = z_positions, labels= zs )
plt.colorbar(shrink=0.18, aspect=16, label=fr'$\rm \delta T_b6e9$'  r'$\rm - \delta T_b1e8$' '\n' '$\;[mK]$')

plt.xlabel('Redshift z')
plt.ylabel('y-axis [Mpc]')
fig.savefig(f'{savedir}LC_diff.pdf', bbox_inches='tight')