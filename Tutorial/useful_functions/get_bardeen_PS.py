import powerbox as pbox
import numpy as np
# Everything in this cell is from 21cmFAST
Om0=0.30964144154550644

# Constants and global parameters
theta_cmb = 2.7255/2.7  # Example: CMB temperature today
N_nu = 3.046 # Number of neutrino species

SIGMA_8=0.8102
hlittle=0.6766
OMm=0.30964144154550644
OMl=0.6889
OMb=0.04897468161869667
POWER_INDEX=0.9665
OMnu=0.0



omhh = OMm*hlittle**2 # Omega_m * h^2, h is dimensionless Hubble constant
# Cosmological parameters (placeholders, specify actual values)
# f_nu = 0.0  # Neutrino density fraction
f_nu=OMnu/OMm

if f_nu<1e-16: f_nu=1e-10
# f_baryon = 0.0486  # Baryon density fraction
f_baryon=OMb/OMm

# CosmoParams structure
cosmo_params_ps = {
    'OMb': OMb,  # Baryon density parameter
    'hlittle': hlittle  # Reduced Hubble constant
}

Radius_8 = 8.0/hlittle
kstart = 1.0e-99/Radius_8
kend = 350.0/Radius_8

lower_limit = kstart
upper_limit = kend



def TFmdm(k):
    q = k * theta_cmb**2 / omhh
    gamma_eff = np.sqrt(alpha_nu) + (1.0 - np.sqrt(alpha_nu)) / (1.0 + np.power(0.43 * k * sound_horizon, 4))
    q_eff = q / gamma_eff
    TF_m = np.log(np.e + 1.84 * beta_c * np.sqrt(alpha_nu) * q_eff)
    TF_m /= TF_m + np.power(q_eff, 2) * (14.4 + 325.0 / (1.0 + 60.5 * np.power(q_eff, 1.11)))
    q_nu = 3.92 * q / np.sqrt(f_nu / N_nu)
    TF_m *= 1.0 + (1.2 * np.power(f_nu, 0.64) * np.power(N_nu, 0.3 + 0.6 * f_nu)) / \
           (np.power(q_nu, -1.6) + np.power(q_nu, 0.8))
    return TF_m

def TFset_parameters():
    global alpha_nu, beta_c, sound_horizon
    z_equality = 25000 * omhh * np.power(theta_cmb, -4) - 1.0
    k_equality = 0.0746 * omhh / (theta_cmb * theta_cmb)

    z_drag = 0.313 * np.power(omhh, -0.419) * (1 + 0.607 * np.power(omhh, 0.674))
    z_drag = 1 + z_drag * np.power(cosmo_params_ps['OMb'] * cosmo_params_ps['hlittle'], 0.238 * np.power(omhh, 0.223))
    z_drag *= 1291 * np.power(omhh, 0.251) / (1 + 0.659 * np.power(omhh, 0.828))

    y_d = (1 + z_equality) / (1.0 + z_drag)

    R_drag = 31.5 * cosmo_params_ps['OMb'] * cosmo_params_ps['hlittle']**2 * np.power(theta_cmb, -4) * 1000 / (1.0 + z_drag)
    R_equality = 31.5 * cosmo_params_ps['OMb'] * cosmo_params_ps['hlittle']**2 * np.power(theta_cmb, -4) * 1000 / (1.0 + z_equality)

    sound_horizon = 2.0 / 3.0 / k_equality * np.sqrt(6.0 / R_equality) * \
                    np.log((np.sqrt(1 + R_drag) + np.sqrt(R_drag + R_equality)) / (1.0 + np.sqrt(R_equality)))

    f_c = 1 - f_nu - f_baryon
    f_cb = 1 - f_nu
    f_nub = f_nu + f_baryon
    p_c = -(5 - np.sqrt(1 + 24 * (1 - f_nu - f_baryon))) / 4.0
    p_cb = -(5 - np.sqrt(1 + 24 * (1 - f_nu))) / 4.0

    alpha_nu = (f_c / f_cb) * (2 * (p_c + p_cb) + 5) / (4 * p_cb + 5.0)
    alpha_nu *= 1 - 0.553 * f_nub + 0.126 * np.power(f_nub, 3)
    alpha_nu /= 1 - 0.193 * np.sqrt(f_nu) + 0.169 * f_nu
    alpha_nu *= np.power(1 + y_d, p_c - p_cb)
    alpha_nu *= 1 + (p_cb - p_c) / 2.0 * (1.0 + 1.0 / (4.0 * p_c + 3.0) / (4.0 * p_cb + 7.0)) / (1.0 + y_d)
    beta_c = 1.0 / (1.0 - 0.949 * f_nub)

TFset_parameters()  # Initialize parameters

# Example usage
k = 0.003  # Wavenumber in h/Mpc
transfer_function_value = TFmdm(k)
# print("Transfer Function at k =", k, "is", transfer_function_value)


def power_in_k(k):
    T = TFmdm(k);
    p = pow(k, POWER_INDEX) * T * T
    return p

def sigma_k2(k, Z='dummy', transfer='dummy'):
    return power_in_k(k)

def power_per_k(k, Z, transfer = 'dummy', box_volume='dummy'):
    return pow(k,3)/(2*pow(np.pi,2)) * sigma_k2(k,Z,transfer)







# This cell is just to find the normalization
from scipy.integrate import quad
def window(space_variable,
           type_ = 'spherical_top_hat', # spherical_top_hat,  sharp_k_space   or   gaussian 
           R = 8/hlittle ,                       # Mpc
           space='k'):           # k or r
    # SPHERICAL TOP HAT--------------------------------------------------------
    if type_=='spherical_top_hat':
        Vw = 4*np.pi/3  * pow(R, 3)
        if space == 'r':
            if abs(space_variable)<=R : return 1
            else                      : return 0 
        if space == 'k': 
            return 4* np.pi * pow(R, 3) * (    np.sin(space_variable*R) / pow(space_variable* R , 3)  - np.cos(space_variable * R) / pow(space_variable* R , 2)          )  
        
def volume_window(type_ = 'spherical_top_hat', # spherical_top_hat,  sharp_k_space   or   gaussian 
           R = 8/hlittle ):                       #  Mpc
    # SPHERICAL TOP HAT--------------------------------------------------------
    if type_=='spherical_top_hat':
        Vw = 4*np.pi/3  * pow(R, 3)
    return Vw

# Radius to mass
def M_corresponding_to_R(R, type_of_window = 'spherical_top_hat'):
    return volume_window(type_ = type_of_window, R = R) * RHO_0 # M_sun
# Mass to Radius
def R_corresponding_to_M(M, type_of_window = 'spherical_top_hat'):
    return pow(   M / ( volume_window(type_ = type_of_window, R = 1.0) * RHO_0 ) ,  1/3.0) # Mpc

SIGMA8=SIGMA_8
# function to integrate:
def integrand(k, 
              R = 8/hlittle, # in Mpc
              Z = 0, 
              type_of_window = 'spherical_top_hat', 
              transfer = 'dummy'):

    Vw = volume_window(type_ = type_of_window, R = R )
    return power_per_k(k, Z=Z , transfer = transfer)     *     pow(window(k, type_ = type_of_window, R = R , space='k'), 2)     *    1/k * pow(Vw, -2)

def sigma_m2(R, Z=0, norm8 = 1, type_of_window = 'spherical_top_hat', transfer = 'dummy'):
    return norm8 * quad(lambda k: integrand(k, R, Z, type_of_window, transfer), 0, np.inf)[0]  # or simpson(f,0.001, 1000, 100000)


norm8 = SIGMA8**2 / sigma_m2(R = 8/hlittle)




BOX_VOLUME=300*300*300

def power_21c(k): # insert k 
    '''Takes in a wavenumber k (in Mpc^-1), returns sigma_k^2 (in Mpc^3), i.e. the power of the power spectrum using the bardeen transer function'''
    return BOX_VOLUME* norm8 * sigma_k2(k, Z='dummy', transfer = 'dummy')

# sample the fourier transform of the density field delta(k) 
def sample_k_21c(k_amplitude):
     return np.sqrt(power_21c(k_amplitude)/2) * complex( np.random.normal(loc=0.0, scale=1.0) , np.random.normal(loc=0.0, scale=1.0))