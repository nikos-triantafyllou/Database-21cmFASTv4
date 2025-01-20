import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from IPython.display import clear_output

try:
    EoR_colour = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap',\
    [(0, 'white'),(0.33, 'yellow'),(0.5, 'orange'),(0.68, 'red'),\
    (0.83333, 'black'),(0.9, 'blue'),(1, 'cyan')])
    matplotlib.colormaps.register(cmap=EoR_colour,name='EoR')
except:
    print('EoR_color already loaded')


    
def plot_lc(tb_array, redshift_arr=0):
    '''
    Function for plotting nice lightcones
    Inputs: 
    tb_array: A shape (x,y,redshift) lightcone np.array
    redshift_arr: an array with redshift values corresponding to the Mpc positions in the lightcone
    Outputs:
    figure
    
    
    '''
    fig, ax = plt.subplots(figsize=((10,8)))
    plt.imshow(tb_array[:,7,], cmap='EoR', vmin = -150, vmax=30, extent=[0,2553,0,300])
    ax.set_aspect(1.5)
    
    if type(redshift_arr)!=int:
        zs= np.concatenate((np.arange(5,15,1), np.arange(15,23,2)))
        zs = np.concatenate((zs,np.arange(23,37,3)))
        zs=[5,6,7,8,9,11,13,16,20,26,35]

        z_positions=[]
        for z in zs:
            z_positions.append( np.where(abs(redshift_arr-z) ==  min(abs(redshift_arr-z)))[0][0] )

        ax.set_xticks(ticks = z_positions, labels= zs )

    plt.colorbar(shrink=0.18, aspect=16, label=r'$\rm \delta T_b \;[mK]$')

    plt.xlabel('Redshift z')
    plt.ylabel('y-axis [Mpc]')
#     plt.close()
    plt.show()

    return fig


def plot_box_slice(box_slice):
    '''
    Function for plotting nice coeval slices
    Inputs: 
    box: A shape (x,y) np.array
    Outputs:
    figure
    '''
    
    fig, ax = plt.subplots()
#     figsize=((10,5)
    
    plt.imshow(box_slice[:,:],cmap='EoR',vmin = -150,vmax=30)
    
    plt.colorbar(shrink=1, aspect=30, label=r'$\rm \delta T_b \;[mK]$')

    plt.xlabel('x-axis [Mpc]')
    plt.ylabel('y-axis [Mpc]')
#     plt.close()
    plt.show()

    return fig


# def play_movie(func,  *args, **kwargs, seed_arr):

    