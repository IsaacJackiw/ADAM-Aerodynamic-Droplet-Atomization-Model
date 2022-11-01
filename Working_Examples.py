# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 08:51:30 2022

@author: Isaac Jackiw


This code provides basic working examples of the models developed in the following works:

   - Jackiw, I. M., and Ashgriz, N., 2021, “On Aerodynamic Droplet Breakup,” J. Fluid Mech., 913(A33). https://doi.org/10.1017/jfm.2021.7
   - Jackiw, I. M., and Ashgriz, N., 2022, “Prediction of the Droplet Size Distribution in Aerodynamic Droplet Breakup,” J. Fluid Mech., 940, p. A17. https://doi.org/10.1017/jfm.2022.249
   - Jackiw, I. M., and Ashgriz, N., "Aerodynamic Droplet Atomization Model (ADAM): A Prediction of the Droplet Size Distribution For Twin-Fluid Nozzles", revision submitted October 25, 2022 (JFM-22-1137.R1)

This code is made available for the purpose of sharing the contained models so that they can be more readily used and improved upon by others.    
Any works that implement or build from this code should cite the above references.

The code was written by the author in Python 3.7.9 using the Spyder 4.2.0 IDE from Anaconda (https://www.anaconda.com/)
    and is not guaranteed to work on any other platform without modification.
    
Note: The author of this code is not a professional program developer.


Make it a great day,

Isaac



"""


import numpy as np

import matplotlib.pyplot as plt




""" Droplet Breakup Model - example implementation """
from Breakup_Model import breakup_distribution


#Conditions from Guildenbecher et al. (2017)

#Fluid properties
rho_l = 789 #kg/m^3
mu_l = 1.2e-3 #Pa.s

rho_g =  1.2 #kg/m^3
mu_g = 1.825e-5
sigma = 2.44e-2 #N/m - Surface Tension


# Droplet conditions
d_0 = 2.54*1e-3 #[mm]
We = 13.8

# d_0 = 2.55*1e-3 #[mm]
# We = 55.33

U = np.sqrt(We*sigma/(d_0*rho_g))


props = rho_l, rho_g, mu_l, mu_g, sigma



sizes, all_dist, d_core, SMD = breakup_distribution(d_0, U, props, Model_Core_Breakup = 1, core_accel = True, dist_func='gamma', unpack =0)


plt.figure()

#Note: Normalized to initial droplet size, d_0
plt.plot(sizes/(d_0*1e6), all_dist*(d_0*1e6), 'k-')

#Plot SMD
plt.axvline(x=SMD/(d_0*1e6), c='r', ls = '-')


plt.xlim(0,1)
plt.ylim(0,)
plt.xlabel('Child droplet size : $d/d_0$')
plt.ylabel('Volume-weighted probability density : $p_v$ ')

plt.tight_layout()







""" Twin-fluid Breakup Model - example implementation """

from Breakup_Model import twin_fluid_breakup_distribution


#Phase properties

rho_l = 1000 #kg/m^3
mu_l = 1e-3 #Pa.s

rho_g =  1.24 #kg/m^3
mu_g = 1.8e-5
sigma = 7.29e-2 #N/m - Surface Tension


props = rho_l, rho_g, mu_l, mu_g, sigma


#Nozzle dimensions

d_go = 1.78e-3  #m
d_gi = 1.27e-3  #m

d_l = d_gi      #m


#Flow conditions
u_g = 116       #m/s
u_l = 0.26      #m/s




SMD, f, bin_edges, bin_widths, bin_centers = twin_fluid_breakup_distribution(d_go, d_gi, u_g, u_l, props,\
                                                                             dist_type = 'vol', freq = True, dist_func = 'lognorm')


plt.figure()

#Plot as line
plt.plot(bin_centers, f, 'k.-')

#Plot as histogram
plt.bar(bin_edges, f, bin_widths, align = 'edge', \
        color='0.1', ec='k', lw=0.5, alpha = 0.5, ls = '-') #, zorder=-10

#Plot SMD
plt.axvline(x=SMD, c='r', ls = '-')


plt.xscale('log')
plt.xlim(1e0,1e3)

plt.ylabel('Volume frequency : $f_v$ (%)')
plt.xlabel('$d$ ($\mu$m)')

plt.tight_layout()
