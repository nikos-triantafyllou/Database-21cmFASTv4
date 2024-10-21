import matplotlib.pyplot as plt
import h5py


filename = '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_simple_run/test_simple_run.h5'
savedir= '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_simple_run/plots/'



with h5py.File(filename, 'r') as hdf:
                node_redshifts = hdf['coeval_data'].attrs['node_redshifts']
                tb_global = hdf['coeval_data']['Tb_z_global'][:]
                xH_global = hdf['coeval_data']['EoR_history'][:]


fig, ax = plt.subplots() 
plt.plot(node_redshifts, tb_global, color = 'black' )
plt.xlabel('redshift, z')
plt.ylabel('Tb')
fig.savefig(f'{savedir}Tb_global.png')


fig, ax = plt.subplots() 
plt.plot(node_redshifts, xH_global, color = 'black' )
plt.xlabel('redshift, z')
plt.ylabel('Tb')
fig.savefig(f'{savedir}xH_global.png')
