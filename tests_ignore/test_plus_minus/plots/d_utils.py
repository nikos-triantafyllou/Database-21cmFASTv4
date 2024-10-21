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