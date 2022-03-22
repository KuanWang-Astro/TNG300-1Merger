from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['mathtext.rm'] = 'serif'
rcParams['mathtext.it'] = 'serif:italic'
rcParams['mathtext.bf'] = 'serif:bold'
rcParams['axes.titlepad'] = 12
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['figure.dpi'] = 80
rcParams['figure.figsize'] = [8, 6]
rcParams['figure.titlesize'] = 25
rcParams['font.size'] = 20.0
rcParams['legend.fontsize'] = 20
rcParams['legend.frameon'] = False
rcParams['savefig.pad_inches'] = 0.1
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

import numpy as np
import collections
from scipy import stats
import itertools
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

###################################
# Cosmological calculations - TNG #
###################################

h = 0.6774
omega_m = 0.3089
omega_L = 1 - omega_m
cosmo = FlatLambdaCDM(H0 = h * 100, Om0 = omega_m)
const1 = 0.102271217 # 100km/s/Mpc = const1 Gyr^-1, H0 = h * const1 Gyr^-1
const2 = 3.2407793E-18 # 100km/s/Mpc = const2 s^-1, H0 = h * const2 s^-1
G = 4.5170908E-48 # grav. constant in s^-2/(Msun/Mpc^3)

def E2(z):
    return omega_m * (1 + z) ** 3 + omega_L

def Delta_c(z):
    x = - omega_L / E2(z)
    return 18 * np.pi * np.pi + 82 * x - 39 * x * x

def rho_c(z): # physical in Msun / Mpc^3
    return 3 * h * h * const2 * const2 * E2(z) / 8 / np.pi / G

def age(a): # in Gyr
    z = 1 / a - 1
    return cosmo.age(z)

###################################
######### Halo properties #########
###################################

def Rvir(Mvir, a): # Mvir at a in Msun/h, Rvir comoving in kpc/h
    z = 1 / a - 1
    return np.power(3 * Mvir / h / 4 / np.pi / Delta_c(z) / rho_c(z),
                    1 / 3) * 1000 * h / a

def Vvir(Mvir, a): # Mvir at a in Msun/h, Vvir physical in km/s
    return np.sqrt(G * Mvir / Rvir(Mvir, a) / a * 1000) * 3.08567758E19

def cv(Mvir, Vmax, a): # Vmax / Vvir
    return Vmax / Vvir(Mvir, a)

###################################
######### Dynamical times #########
###################################

a_list = np.load('a_list.npy')

def tdyn(a): # in Gyr
    z = 1 / a - 1
    return np.pi / h / const1 / np.sqrt(E2(z) * 2 * Delta_c(z))

def ntdyn(a, ai = a_list[0]): # number of tdyn's at a since ai
    z = 1 / a - 1
    zi = 1 / ai - 1
    return quad(lambda x: 1 / (1 + x) / tdyn(1 / (1 + x)) / np.sqrt(E2(x)),
                z, zi)[0] / h / const1

a_interp = np.linspace(a_list[0], 1, 1000)
n_interp = np.array([ntdyn(a) for a in a_interp])
ntau_a = interp1d(a_interp, n_interp)
a_ntau = interp1d(n_interp, a_interp)

###################################
####### Time steps in tree ########
###################################

z_list = 1 / a_list - 1
n_list = ntau_a(a_list)


###################################
########## Major mergers ##########
###################################

def a_before_ai(ai, n_back = 0.25):
    ni = ntau_a(ai)
    n_before = ni - n_back
    assert n_before >= 0
    return a_ntau(n_before)

def a_after_ai(ai, n_forth = 0.25):
    ni = ntau_a(ai)
    n_after = ni + n_forth
    assert n_after <= ntau_a(1)
    return a_ntau(n_after)

def Gamma(Mnext, Mlast, anext, alast):
    dntau = ntau_a(anext) - ntau_a(alast)
    return (Mnext - Mlast) / Mlast / dntau

def prophist_z_interp(prophist): #prop can be logM for example
    mask = np.isnan(prophist)
    return interp1d(z_list[~mask], prophist[~mask])

def prophist_a_interp(prophist):
    mask = np.isnan(prophist)
    return interp1d(a_list[~mask], prophist[~mask])
    
def Gamma_interpolated(logMhist, ai, n_back = 0.25, n_forth = 0.25):
    logM_z = prophist_z_interp(logMhist)
    M_before = np.power(10, logM_z(1 / a_before_ai(ai, n_back = n_back) - 1))
    M_after = np.power(10, logM_z(1 / a_after_ai(ai, n_forth = n_forth) - 1))
    return M_after / M_before - 1, (M_after / M_before - 1) / (n_back + n_forth)
