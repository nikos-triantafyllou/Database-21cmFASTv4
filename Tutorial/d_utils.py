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





import h5py, glob, os
# Some useful functions
def check_file_exists(savedir, filename):
    """
    Check if a file exists.

    Parameters:
    savedir (str): The path to the file
    filename (str): The name of the file.

    Returns:
    bool: True if the file exists, False otherwise.
    """
    return os.path.isfile(filename)


def print_colored_message(message, color):
    """
    Print a message in a specified color.

    Parameters:
    message (str): The message to print.
    color (str): The color code (e.g., 'green' or 'red').
    """
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow':'\033[93m', 
        'endc': '\033[0m',
    }
    print(f"{colors[color]}{message}{colors['endc']}")

# # Example usage:
# filename = 'example.h5'
# if check_file_exists(savedir, filename):
#     print_colored_message(f"The file '{filename}' already exists.", 'red')
# else:
#     print_colored_message(f"The file '{filename}' does not exist.", 'green')

    
    
def create_group_with_attributes(parent_group, group_name, params_dict):
    """
    create a group within a hdf5 file

    Parameters:
    parent_group (str).
    group name (str).
    params_dict (dict): dictionary with the parameters to save as attributes (NOTE: skips keys that have None values).
    """
    group = parent_group.create_group(group_name)
    for key, value in params_dict.items():
        if value is not None:
            group.attrs[key] = value
            

        
        
        
def pkat(group_or_dataset, indent='', returnn=False): # Print Keys, ATtributes 
    '''
    Print keys and attributes of a specific file, group or dataset within a hdf5 file
    
    Parameters:
    group_or_dataset
    
    Example usage:
    
    file = h5py.File('example.h5', 'r') 
    pkat(file)
    
    or 
    
    pkat(file['optional group'][optional_dataset])
    '''
    try:
        print(indent,'Group with attrs:' , dict(group_or_dataset.attrs) , '\n'+indent, 'with keys:', list(group_or_dataset.keys()))
        if returnn==True:
            return list(group_or_dataset.keys())
    except:
        print(indent,'Dataset with attrs:' , dict(group_or_dataset.attrs), '\n'+indent, 'Dataset:', group_or_dataset )
        
        
def explore_hdf5(filename):
    gap='       '
    with h5py.File(filename, 'r') as ff:
        print_colored_message(filename, 'red')
        keys = pkat(ff,'', returnn=True)
#         print('go dowm')
        for key in keys:
            print_colored_message(gap+key, 'yellow')
            keys2 = pkat(ff[key], gap, returnn=True)
            for key2 in keys2:
#                 print(gap+gap+key2)
                print_colored_message(gap+gap+key2, 'green')
                keys3 = pkat(ff[key][key2],gap+gap, returnn=True)
                
                
# read the cache file names
def search_files(folder, pattern):
    '''
    Parameters: 
    Search for a name pattern in a specific folder
    folder (str)
    pattern (str)
    
    Returns:
    List with all the names of the files (strings)
    '''
    matches = glob.glob(os.path.join(folder, '**', pattern), recursive=True)
    file_names = [os.path.basename(match) for match in matches]  # Extract only file names
    return file_names





# Database only-------------------------------------------------------------

def is_xHI_flagged(x_HI_coev):
    '''
    Function to see if the neutral gas fraction evolution from 21cmFAST is weird or not
    Parameters: 
    numpy array with DESCENDING values for xHI, i.e. high to low redshit, i.e. from past to present
    
    Returns:
    flag which is either True of False (True = weird result)
    '''
    flag = False
    strictness = 1e-4
    for i in range(1, len(x_HI_coev), 1):
#         print(x_HI_coev[i-1]-x_HI_coev[i])
        if x_HI_coev[i-1]-x_HI_coev[i]<-strictness:
            flag = True
        if i >=2 and x_HI_coev[i-2]-x_HI_coev[i]<-strictness:
            flag = True
        if i >=3 and x_HI_coev[i-3]-x_HI_coev[i]<-strictness:
            flag = True
        if i >=4 and x_HI_coev[i-4]-x_HI_coev[i]<-strictness:
            flag = True
        if i >=5 and x_HI_coev[i-5]-x_HI_coev[i]<-strictness:
            flag = True
    return flag




import d_get_properties

def get_uvlf(redshift, opened_hdf5):
    hdf = opened_hdf5
    # Format the parameters 
    user_params = dict(hdf['simulation_parameters']['user_params'].attrs)
    cosmo_params = dict(hdf['simulation_parameters']['cosmo_params'].attrs)
    astro_params = dict(hdf['simulation_parameters']['astro_params'].attrs)
    flag_options = dict(hdf['simulation_parameters']['flag_options'].attrs)


    input_params_struct = p21c.InputParameters(
                    cosmo_params=p21c.CosmoParams.new(cosmo_params),
                    astro_params=p21c.AstroParams.new(astro_params),
                    user_params=p21c.UserParams.new(user_params),
                    flag_options=p21c.FlagOptions.new(flag_options),
                    random_seed=hdf.attrs['random_seed'],
                    node_redshifts = hdf['coeval_data'].attrs['node_redshifts'],
                )
    
    
    
    # Get the correct redshift data
    halo_data_redshifts = list(hdf['halo_data'].keys())
    for i in range(len(halo_data_redshifts)):
        if abs(redshift - float(halo_data_redshifts[i]) )<0.1:
            correct_redshift_key_str = halo_data_redshifts[i]
        
    hdf_group = hdf['halo_data'][correct_redshift_key_str]
    total_number_of_halos = len(hdf_group['halo_masses'][:])
#     n_halos=1000000
    n_halos = total_number_of_halos
    
    
    
    hm_cat, sm_cat, sfr_cat = d_get_properties.get_properties(redshift     = correct_redshift_key_str, 
                                                          n_halos      = n_halos, 
                                                          random_seed  = hdf.attrs['random_seed'], 
                                                          hdf_group    = hdf_group, 
                                                          cosmo_params = p21c.CosmoParams.new(dict(hdf['simulation_parameters']['cosmo_params'].attrs)), 
                                                          user_params  = p21c.UserParams.new(dict(hdf['simulation_parameters']['user_params'].attrs)), 
                                                          flag_options = p21c.FlagOptions.new(dict(hdf['simulation_parameters']['flag_options'].attrs)), 
                                                          astro_params = p21c.AstroParams.new(dict(hdf['simulation_parameters']['astro_params'].attrs)))
    
    # Relations used also in 21cmFAST (so also for the previous UVLFs)
    Luv_over_SFR = 1.0 / 1.15 / 1e-28 # [M_solar yr^-1/erg s^-1 Hz^-1], G. Sun and S. R. Furlanetto (2016) MNRAS, 417, 33)
    Muv_cat = 51.63 - 2.5*np.log10(sfr_cat*Luv_over_SFR)

    hist, bin_edges = np.histogram(Muv_cat, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[3]-bin_edges[2]

    # Normalize the histogram 1. by the survey volume, 2. bin width (for /mag/Mpc^3) and 3. by the number of sampled halos out of the total number 
    sim_volume= 300*300*300
    hist = hist/ (bin_width * sim_volume)
    hist*= total_number_of_halos/n_halos
    
    return bin_centers, hist