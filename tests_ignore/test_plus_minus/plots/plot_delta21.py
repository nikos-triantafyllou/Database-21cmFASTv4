# PARAMETERS-------------------------------------------
filenames = ['/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_simple_run/test_simple_run.h5']
redshift_colorbar = False # If True in one plot all of the redshifts will be plotted 
diff=False
plot_single=True
# filenames = ['/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.01_r23.h5',
#              '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.0166810054_r23.h5',
#             '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.0774263683_r23.h5',
#              '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.215443469_r23.h5',
#             #  '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.027825594_r23.h5',
#              '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.359381366_r23.h5',
#             #  '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.0464158883_r23.h5',
#              '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_min_sampler_mass/smm_0.59948425_r23.h5'
# ]

savedir= '/leonardo_work/CNHPC_1497299/ntriantafyllou/database/database3_venv/test_simple_run/plots/'

#-----------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import h5py
import d_plotting
import py21cmfast_tools as p21c_tools

print('imported stuff', flush=True)

if redshift_colorbar:
    for filename in filenames:
        with h5py.File(filename, 'r') as hdf:
                lightcone_redshifts = hdf['lightcones'].attrs['lightcone_redshifts']
                lc = hdf['lightcones']['brightness_temp'][:]
                smm= "%.2e" % dict(hdf['simulation_parameters']['user_params'].attrs)['SAMPLER_MIN_MASS']

        print(f'read files,{smm}', flush=True)

        ps = p21c_tools.calculate_ps(lc=lc,
                        lc_redshifts = lightcone_redshifts,
                        box_length = 300.0,
                        #    box_side_shape = 200,
                        calc_1d = True,
                        calc_2d = False,
                        )
        

    #     d_plotting.plot_ps_1D(ps,smm)
        fig = d_plotting.plot_ps_1D_2(ps,smm)
        fig.savefig(f'{savedir}delta21.pdf')

if diff:
    ps_dict = {}
    for filename in filenames:
        with h5py.File(filename, 'r') as hdf:
                lightcone_redshifts = hdf['lightcones'].attrs['lightcone_redshifts']
                lc = hdf['lightcones']['brightness_temp'][:]
                smm= "%.2e" % dict(hdf['simulation_parameters']['user_params'].attrs)['SAMPLER_MIN_MASS']
        ps=p21c_tools.calculate_ps(lc=lc,
                        lc_redshifts = lightcone_redshifts,
                        box_length = 300.0,
                        box_side_shape = 200,
                        calc_1d = True,
                        calc_2d = False,
                        )
        ps_dict[f'{filename}']=ps


    which_k = 1
    title = ps_dict[filenames[0]]['k'][which_k]



    xarr = ps_dict[filenames[0]]['redshifts']
    fiducial_yarr = ps_dict[filenames[0]]['ps_1D'][:, which_k]
    ylabel = 'power'
    xlabel = 'z'
    fiducial_label = '1.00e8'


    # for array_of_yarrs
    array_of_yarrs = []
    for filename in filenames[1:]:
        array_of_yarrs.append(ps_dict[filename]['ps_1D'][:,which_k])

    array_of_other_labels = ['plus', 'minus']


    fig=d_plotting.plot_with_diff(xarr, fiducial_yarr, array_of_yarrs , fiducial_label = fiducial_label, array_of_other_labels=array_of_other_labels, ylabel=ylabel, xlabel=xlabel, title=title)
    fig.savefig(f'{savedir}delta21_diff_{which_k}.pdf')


if plot_single:
    for filename in filenames:
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
    which_k = 9
    title = ps['k'][which_k]

    xarr = ps['redshifts']
    yarr = ps['ps_1D'][:, which_k]
    ylabel = 'power'
    xlabel = 'z'


    fig, ax = plt.subplots()
    ax.plot(xarr, yarr, color='black', alpha=1)


    # ax.plot(xarr, array_of_yarrs[i], label = array_of_other_labels[i], color=colors[i], alpha=1)
    ax.grid()
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    # ax.legend()
    ax.set_yscale('log')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_title(label=f'k={title}')

    fig.savefig(f'{savedir}delta21_{which_k}.pdf')