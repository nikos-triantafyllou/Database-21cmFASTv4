import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap

try:
    EoR_colour = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap',\
    [(0, 'white'),(0.33, 'yellow'),(0.5, 'orange'),(0.68, 'red'),\
    (0.83333, 'black'),(0.9, 'blue'),(1, 'cyan')])
    matplotlib.colormaps.register(cmap=EoR_colour,name='EoR')
except:
    print('EoR_color already loaded')


def plot_hmf_z(HMF_arr, hist_mass_mean, node_redshifts, title='HMF' ):
    '''
    Plots the Halo Mass Function (HMF) as is was saved in the HDF5 file as a function of redshift 
    
    Parameters:
    HMF_arr: 2D numpy array with the values of the HMF in different mass bins and redshifts (shape: (len(node_redshifts), len(hist_mass_mean)))
    hist_mass_mean: mean masses of the mass bins in the halo mass histogram
    node_redshifts: node redshifts of the coeval boxes in 21cmFAST 
    title: title of the plot
    
    Returns:
    Figure: dn/dlnm [???] vs halo mass [???] vs z(color)
    '''
    fig, ax = plt.subplots()

    # plot sigma_k2 for as many redshifts indicated by n_plots
    n_plots = 92  # Number of times I want to plot the function
    colormap = plt.cm.viridis_r  
    colorlist = [colormap(i) for i in np.linspace(0, 1, n_plots)]

    redshifts= node_redshifts
    color=0
    for redshift in redshifts:
        ax.plot(hist_mass_mean, HMF_arr[color], color=colorlist[color], alpha=1)
        color+=1

    # # Plot the value of the scale of matter-radiation equality
    # plt.axvline(k_eq, color='black', linestyle='--', label=r'$\rm k_{eq}=$'+'{}'.format(round(k_eq,3)) + r'$\rm h Mpc^{-1}$')

    # Make the plot pretty
    plt.xlabel(r'$\rm M_{\rm halo}~[M_\odot]$')
    plt.ylabel(r'$\rm dn/dlnM~[Mpc^{-3}]$')
    plt.title(label=title)
    plt.xscale('log')
    plt.yscale('log')

    # plt.legend()
    plt.grid()

    # Add a colorbar for the different redshift plots
    norm = Normalize(vmin=redshifts[-1], vmax=redshifts[0])
    sm = ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    plt.colorbar(sm, ax=ax, ticks=np.linspace(35, 5, 12), label='Redshift, z', aspect=25)
    # plt.savefig()
    plt.close()

    return fig


def plot_lc(lc, lc_redshifts, title = 'lightcone', vmin = -150, vmax=30 ):
    '''
    Plot a lightcone
    
    Parameters:
    ----------
    lc: lightcone array, shape (len_box, len_box, len_redshifts)
    lc_redshifts: redhsifts
    title: title of the plot
    
    Outputs:
    --------
    figure
    
    '''
    lightcone = lc
    redshift_arr = lc_redshifts
    
    fig, ax = plt.subplots(figsize=((10,8)))
    plt.imshow(lightcone[:,123,:],cmap='EoR',vmin = vmin,vmax=vmax, extent=[0,2553,0,300])
    ax.set_aspect(1.5)

    zs= np.concatenate((np.arange(5,15,1), np.arange(15,23,2)))
    zs = np.concatenate((zs,np.arange(23,37,3)))
    zs=[5,6,7,8,9,11,13,16,20,26,35]

    z_positions=[]
    for z in zs:
        z_positions.append( np.where(abs(redshift_arr-z) ==  min(abs(redshift_arr-z)))[0][0] )

    ax.set_xticks(ticks = z_positions, labels= zs )
    # ax.set_yticks(ticks = z_positions, labels= zs )
    plt.colorbar(shrink=0.18, aspect=16, label=r'$\rm \delta T_b \;[mK]$')

    plt.xlabel('Redshift z')
    plt.ylabel('y-axis [Mpc]')
    plt.title(title)
#     plt.close()
    return fig 



def plot_ps_1D(ps,title='power vs k'):
    fig, ax = plt.subplots()

    # plot sigma_k2 for as many redshifts indicated by n_plots
    n_plots = 64  # Number of times I want to plot the function
    colormap = plt.cm.viridis_r  
    colorlist = [colormap(i) for i in np.linspace(0, 1, n_plots)]

    redshifts= ps['redshifts']
    color=0
    for redshift in redshifts:
        ax.plot(ps['k'][1:], ps['ps_1D'][color,1:], color=colorlist[color], alpha=1)
        color+=1

    # # Plot the value of the scale of matter-radiation equality
    # plt.axvline(k_eq, color='black', linestyle='--', label=r'$\rm k_{eq}=$'+'{}'.format(round(k_eq,3)) + r'$\rm h Mpc^{-1}$')

    # Make the plot pretty
    plt.xlabel(r'$\rm k$')
    plt.ylabel(r'$\rm power$')
    # plt.title(label=title)
    plt.xscale('log')
    plt.yscale('log')

    # plt.legend()
    plt.grid()

    # Add a colorbar for the different redshift plots
    norm = Normalize(vmin=redshifts[-1], vmax=redshifts[0])
    sm = ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    plt.colorbar(sm, ax=ax, ticks=np.linspace(35, 5, 12), label='Redshift, z', aspect=25)
    plt.title(title)
    
    
def plot_ps_1D_2(ps,title='power vs z'):    
    fig, ax = plt.subplots()

    # plot sigma_k2 for as many redshifts indicated by n_plots
    n_plots = 14  # Number of times I want to plot the function
    colormap = plt.cm.viridis  
    colorlist = [colormap(i) for i in np.linspace(0, 1, n_plots)]

    redshifts= ps['k'][1:]
    color=0
    for redshift in redshifts:
        ax.plot(ps['redshifts'], ps['ps_1D'][:,color+1], color=colorlist[color], alpha=1)
        color+=1

    # # Plot the value of the scale of matter-radiation equality
    # plt.axvline(k_eq, color='black', linestyle='--', label=r'$\rm k_{eq}=$'+'{}'.format(round(k_eq,3)) + r'$\rm h Mpc^{-1}$')

    # Make the plot pretty
    plt.xlabel(r'$\rm z$')
    plt.ylabel(r'$\rm power$')
    # plt.title(label=title)
    # plt.xscale('log')
    plt.yscale('log')

    # plt.legend()
    plt.grid()

    # Add a colorbar for the different redshift plots
    norm = Normalize(vmin=redshifts[-1], vmax=redshifts[0])
    sm = ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    plt.colorbar(sm, ax=ax, ticks=np.linspace(3, 0.01, 20), label='k', aspect=25)
    plt.title(title)
    
    return fig
    
    
    
def plot_with_diff(xarr, fiducial_yarr, array_of_yarrs , fiducial_label, array_of_other_labels, ylabel='y', xlabel='x', title=''):
    '''
    Plot lines and plot the relative difference with the fiducial in a bottom panel
    
    Parameters
    ----------
    xarr (np.array): x array for all of the lines
    fiducial_yarr (np.array): values from the fiducial line
    fiducial_label (str) : title to go in the legend
    array_of_other_labels (list or np.array): array of the rest of the titles
    
    Output:
    ------
    fig
    
    '''
    
    colors = ['teal', 'tab:red', 'purple', 'orange', 'green']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(xarr, fiducial_yarr, label=fiducial_label, color='black', alpha=1)

    for i in range(len(array_of_yarrs)):
        ax1.plot(xarr, array_of_yarrs[i], label = array_of_other_labels[i], color=colors[i], alpha=1)
    
    
    ax1.grid()
    ax1.set_ylabel(ylabel)
    ax1.legend()
    # ax1.set_yscale('log')
    ax1.tick_params(which='both', direction='in', top=True, right=True)
    ax1.set_title(label=f'{title}')
    
    #  Difference between the functions
    for i in range(len(array_of_yarrs)):
        difference = abs(array_of_yarrs[i] - fiducial_yarr)/ fiducial_yarr
        ax2.plot(xarr, difference, label=array_of_other_labels[i], color=colors[i])
        
        
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Fractional Difference')
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add a horizontal line at y=0
    ax2.tick_params(which='both', direction='in', top=True, right=True)
#     ax2.set_xscale('log')

    # Make the plot pretty
    plt.tight_layout()
    plt.grid()
    # plt.show()
    return fig