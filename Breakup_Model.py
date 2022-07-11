# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:17:11 2021

@author: Isaac Jackiw


This code provides the models developed in the following works:
    - Jackiw, I. M., and Ashgriz, N., 2021, “On Aerodynamic Droplet Breakup,” J. Fluid Mech., 913(A33). https://doi.org/10.1017/jfm.2021.7
    - Jackiw, I. M., and Ashgriz, N., 2022, “Prediction of the Droplet Size Distribution in Aerodynamic Droplet Breakup,” J. Fluid Mech., 940, p. A17. https://doi.org/10.1017/jfm.2022.249
    - Jackiw, I.M., and Ashgriz, N., "Prediction of the Droplet Size Distribution From Twin-Fluid Sprays", submitted July 11, 2022

This code is made available for the purpose of sharing the contained models so that they can be more readily used and improved upon by others.    
Any works that implement or build from this code should cite the above references.

The code was written by the author in Python 3.7.9 using the Spyder 4.2.0 IDE from Anaconda (https://www.anaconda.com/)
    and is not guaranteed to work on any other platform without modification.
    
Note: The author of this code is not a professional program developer.


Make it a great day,

Isaac

"""


import numpy as np

from scipy.stats import gamma , lognorm
import scipy.special as sc

from statsmodels.stats.weightstats import DescrStatsW

from scipy.integrate import simps

import matplotlib.pyplot as plt



""" Droplet deformation models from Jackiw & Ashgriz (2021)
            w/ modifications from Jackiw & Ashgriz (2022) ................."""

def dRdt(d_0, We, tau):
    t_tau = 1/8
    a = 6
    return (d_0/2)/tau*(a/2)**2*(1-128/a**2/We) * t_tau


def di_d0(d_0, We):
    a, b, c = [ 1.63, -1.25,  0.312]
    return a - (a-b)*np.exp(-c*We)


def hi_d0(d_0, We, tau,  props):
    rho_l, rho_g, mu_l, mu_g, sigma = props

    if We < 80:
       return 4/(rho_l*dRdt(d_0, We, tau)**2*d_0/sigma + 5 * di_d0(d_0, We)**2 - 4*1/di_d0(d_0, We)) - 0.05
    else:
        We_rim = rho_l * dRdt(d_0, We, tau)**2 *d_0 / sigma
        return 2/We_rim


def V_disk(R, h):
    return 3/2 * ((2*R)**2*h - 2*(1-np.pi/4)*(2*R)*h**2) * (4/3 * np.pi * (1/2)**3)

def V_rim(R, h):
    return 3/2 * np.pi * (2*R *h**2 - h**3) * (4/3 * np.pi * (1/2)**3)


def t_b(d_0, We, tau, C,  props):

    rho_l, rho_g, mu_l, mu_g, sigma = props

    d_d0 = di_d0(d_0, We)
    h_d0 = hi_d0(d_0, We, tau,  props)

    R_i = d_d0/2*d_0
    h_i = h_d0*d_0

    Vb_V0 = (V_disk(R_i, h_i) - V_rim(R_i, h_i)) / (4/3 * np.pi * (d_0/2)**3)

    A = dRdt(d_0, We, tau)/(d_0/2)
    B = 2*R_i/d_0 - 2* h_i/d_0

    return B/A * (-1 + np.sqrt(1 + 8*tau*C*Vb_V0/np.sqrt(3*We)*A/B**2)) / tau


def Beta_d0(d_0, We, tau,  props):

    rho_l, rho_g, mu_l, mu_g, sigma = props

    d_d0 = di_d0(d_0, We)
    h_d0 = hi_d0(d_0, We, tau,  props)

    R_i = d_d0/2*d_0
    h_i = h_d0*d_0

    Vb_V0 = (V_disk(R_i, h_i) - V_rim(R_i, h_i)) / (4/3 * np.pi * (d_0/2)**3)

    A = dRdt(d_0, We, tau)/(d_0/2)
    B = 2*R_i/d_0 - 2* h_i/d_0

    C = 9.4
    t = t_b(d_0, We, tau, C,  props)*tau

    return 3/4 * 1/Vb_V0 / tau**2 * (B**2*t**2/2 + A*B*t**3/3 + A**2*t**4/12)






""" BREAKUP MODEL from Jackiw & Ashgriz (2022) ............................"""

def single_breakup(d_0, We, tau, w,  props):
    """
    Calculates the characteristic breakup sizes of a droplet using the
    distribution model of Jackiw & Ashgriz (2022). w is the volume-weight of
    the breakup event with respect to the initial droplet.
    
    Inputs:
        - d_0 : droplet diameter
        - We : Weber number of the droplet
        - tau : characterisitic time of the droplet
        - w : volume-weight of the breakup event with respect to the initial
                droplet
        - props : list of fluid properties, as
                    props = rho_l, rho_g, mu_l, mu_g, sigma

                        rho_l (kg/m^3)
                        rho_g (kg/m^3)
                        mu_l (Pa.s)
                        mu_g (Pa.s)
                        sigma (N/m)
        
    Outputs:
        weights, sizes
            - weights = w_nodes, w_rim, w_bag
            - sizes = node_sizes, rim_sizes, bag_sizes
    """

    rho_l, rho_g, mu_l, mu_g, sigma = props

    """ Volume weightings """

    d_d0 = di_d0(d_0, We)
    h_d0 = hi_d0(d_0, We, tau,  props)

    R_i = d_d0/2*d_0
    h_i = h_d0*d_0


    Vd_V0 = V_disk(R_i, h_i) / (4/3 * np.pi * (d_0/2)**3)

    VN_V0 = 0.40 * Vd_V0    #Nodes
    Vr_V0 = V_rim(R_i, h_i) / (4/3 * np.pi * (d_0/2)**3)    #Rim
    Vb_V0 = Vd_V0 - Vr_V0 - VN_V0   #Bag
    Vc_V0 = 1 - Vd_V0   #Core

    w_nodes, w_rim, w_bag, w_core = np.array([VN_V0, Vr_V0, Vb_V0, Vc_V0])


    """ Nodes """

    Dmax_D0 = 2/(1 + np.exp(-0.0019 * We**2.7)) #Zhao2010 correlation
    C_d = 1.2
    lambda_RT = 4*np.pi/np.sqrt(C_d*We) / Dmax_D0

    node_sizes = np.array( [ (3/2*hi_d0(d_0, We, tau,  props)**2*lambda_RT*n)**(1/3)*d_0 \
                    for n in [1, 0.4, 0.2] ] )


    """ Rim breakup """

    #Rim formation

    #Simplified model
    # df_di = 0.64
    # D_l = hi_d0(d_0, We, tau)*df_di * d_0
    # df_d0 = Beta_d0(d_0, We, tau)
    # D_l = hi_d0(d_0, We, tau) * np.sqrt(di_d0(d_0, We) / df_d0) * d_0

    if w_core < 0.1: #Bag
        df_d0 = Beta_d0(d_0, We, tau,  props)
        D_l = hi_d0(d_0, We, tau,  props) * np.sqrt(di_d0(d_0, We) / df_d0) * d_0

    elif (w_core > 0.1) and (w_core < 0.5): #BS
        df_d0 = 2*Beta_d0(d_0, We, tau,  props)
        D_l = hi_d0(d_0, We, tau,  props) * np.sqrt(di_d0(d_0, We) / df_d0) * d_0

    elif (w_core > 0.5) and (We < 80):  #MB
        df_d0 = di_d0(d_0, We)+2*Beta_d0(d_0, We, tau,  props)
        D_l = hi_d0(d_0, We, tau,  props) * np.sqrt(di_d0(d_0, We) / df_d0) * d_0

    elif We > 80: #ST
        D_l = hi_d0(d_0, We, tau,  props) * d_0



    #Bag dynamics
    h = 2.3 * 1e-6 #mean
    # h_std = 1.2 * 1e-6
    V = np.sqrt(2 * sigma / (rho_l * h))
    R = Beta_d0(d_0, We, tau,  props)*d_0
    a_c = V**2/R # Lhuissier 2012 - cetripedal with TC
    b = np.sqrt(sigma/rho_l/a_c) #Bo = 1, Wang2018
    lambda_pred = 4.45*b# *1e3 #RP, Wang2018

    #Collision
    d_c = (3/2 * D_l**2 * lambda_pred)**(1/3)
    #Satellites
    X = mu_l / np.sqrt(rho_l * D_l * sigma)
    d_s_c = d_c/np.sqrt(2+3*X/np.sqrt(2))

    #RP
    d_RP = 1.89*D_l

    #Satellites
    X = mu_l / np.sqrt(rho_l * D_l * sigma)
    d_s_RP = d_RP/np.sqrt(2+3*X/np.sqrt(2))

    #sizes
    rim_sizes = np.array([d_c, d_s_c, d_RP, d_s_RP])


    """ Bag breakup """

    #RP
    d_b_RP = 1.89*b

    #RP Satellites
    X = mu_l / np.sqrt(rho_l * b * sigma)
    d_s = d_b_RP/np.sqrt(2+3*X/np.sqrt(2))

    #sizes
    bag_sizes = np.array([h, d_s, d_b_RP, b])


    """" Returns weights and (absolute) sizes """
    weights = [w_nodes*w, w_rim*w, w_bag*w]#, w_core
    #*w : Adjust the core breakup weights to be relative to the whole drop
    sizes = [node_sizes, rim_sizes, bag_sizes]#, d_core

    return weights, sizes





def check_core_props(d_0, We, tau,  props):
    rho_l, rho_g, mu_l, mu_g, sigma = props

    d_d0 = di_d0(d_0, We)
    h_d0 = hi_d0(d_0, We, tau,  props)

    V_0 = (4/3*np.pi*(d_0/2)**3)

    V_core = V_0 - V_disk(d_d0/2*d_0, h_d0*d_0)

    return V_core > 0


def core_props(d_0, We, tau, U, U_1,  props, core_accel = False):
    """ Calculate core parameters """

    rho_l, rho_g, mu_l, mu_g, sigma = props

    d_d0 = di_d0(d_0, We)
    h_d0 = hi_d0(d_0, We, tau,  props)

    V_0 = (4/3*np.pi*(d_0/2)**3)
    V_core = V_0 - V_disk(d_d0/2*d_0, h_d0*d_0)

    d_core = (V_core * 6/np.pi)**(1/3)

    if core_accel == True:
        C_D = (1.9*0.666 + 6.2*0.333) #Weighted from Chou1997

        C = 9.4
        T_i = 0.9 +  t_b(d_0, We, tau, C,  props)
        U_2 = U_1 + 3/4 * C_D * T_i * (d_d0)**2 * d_0/tau


    else:
        U_2 = U_1

    U_r = U - U_2

    We_core = rho_g*U_r**2*d_core/sigma
    tau_core = d_core/U_r*np.sqrt(rho_l/rho_g)

    return d_core, We_core, tau_core, V_core/V_0, U_2




def n_distribution(mean, std, sizes, dist_func = 'lognorm'):

    if dist_func == 'lognorm':
        #Lognormal
        mean_dist = np.log(mean**2 / np.sqrt(mean**2 + std**2))
        std_dist =np.sqrt(np.log(1+std**2/mean**2))
        median = np.exp(mean_dist)  #This is because of how Python handles the median
        dist = lognorm.pdf(sizes,std_dist, 0, median)

    elif dist_func == 'gamma':
        # Gamma
        a=(mean/std)**2
        Beta = std**2 / mean
        dist = gamma.pdf(sizes, a, 0, Beta)

    return dist


def v_distribution(mean, std, sizes, dist_func = 'lognorm'):

    if dist_func == 'lognorm':
        #Lognormal
        mean_dist = np.log(mean**2 / np.sqrt(mean**2 + std**2))
        std_dist =np.sqrt(np.log(1+std**2/mean**2))
        median = np.exp(mean_dist)  #This is because of how Python handles the median
        dist = lognorm.pdf(sizes,std_dist, 0, median)*(sizes**3)/np.exp(3*np.log(median)+4.5*std_dist**2) #Volume weight & renormalize

    elif dist_func == 'gamma':
        # Gamma
        a=(mean/std)**2
        Beta = std**2 / mean
        dist = gamma.pdf(sizes, a, 0, Beta)*(sizes**3) / (Beta**3 * sc.gamma(a+3) / sc.gamma(a)) #Volume weight & renormalize

    return dist





def breakup_distribution(d_0, U, props, \
                         Model_Core_Breakup = True, core_accel = False, \
                         dist_func = 'lognorm', dist_type = 'vol', unpack = 1):

    """
    Droplet breakup model from Jackiw & Ashgriz (2021, 2022)
        Inputs:
            - d_0 : initial droplet size in m
            - U : relative velocity between gas and droplet in m/s
            - props : list of fluid properties, as
                    props = rho_l, rho_g, mu_l, mu_g, sigma

                        rho_l (kg/m^3)
                        rho_g (kg/m^3)
                        mu_l (Pa.s)
                        mu_g (Pa.s)
                        sigma (N/m)

        Outputs:
            sizes, dist, d_core, d_32
                sizes : array of sizes (x-axis of distribution)
                            if unpack = 0, returns sizes normalized by d_0
                            if unpack = 1, returns absolute sizes in mu m
                dist : distribution (y-axis of distribution)
                            if unpack = 0, unitless (since 'sizes' normalized)
                            if unpack = 1, 1/ mu m
                d_core : size of last unbroken core in mm
                d_32 : d_32 of the number distribution.


        kwargs:
            - Model_Core_Breakup : if TRUE, model breakup of undeformed core.
                                   can also input an INTEGER to model a finite
                                       number of core breakups.
                                   if FALSE, core breakup is not modelled.
            - core_accel : if TRUE, acceleration of core is modelled. if FALSE,
                            core droplet relative velocity is assumed to be U.
            - dist_func : either 'lognorm' or 'gamma'
            - dist_type : 'vol' for volume-weighted probability density
                            or 'num' for number probability density.
            - unpack : how to group characteristic sizes to predict distribution.
                        if 0, each mode of each breakup event is modelled separately.
                        if 1, all sizes are lumped as a single distribution.

    """

    rho_l, rho_g, mu_l, mu_g, sigma = props

    #Calculate relevant non-dim parameters
    We = rho_g*U**2*d_0/sigma
    tau = d_0/U*np.sqrt(rho_l/rho_g)


    #Compute first breakup
    all_breakup = []
    breakup_weights = [1]
    #breakup_weights is used to make sure that the weights are computed relative
    #   to the whole droplet; mainly, for core breakup where only a fraction
    #   (w_c) of the total mass contributes to the event.
    all_breakup.append(single_breakup(d_0, We, tau, breakup_weights[0],  props))


    #Core breakup(s)
    #Compute core conditions
    d_core, We_core, tau_core, w_c, U_core = core_props(d_0, We, tau, U, 0,  props, core_accel)
    core_droplet_sizes = [d_core] #Track core droplet sizes
    core_droplet_velocity = [U_core]
    breakup_weights.append(w_c)

    n=0 #Number of core breakup events

    if Model_Core_Breakup == True:

        if type(Model_Core_Breakup) == int:     #lets you specify how many core breakup events to model
            for i in range(Model_Core_Breakup):
                if (We_core > 8.8) and check_core_props(d_core, We_core, tau_core,  props):
                    n+=1    #Count core breakup event
                    all_breakup.append(single_breakup(d_core, We_core, tau_core, breakup_weights[n],  props)) #Compute core

                    #Compute new core conditions
                    d_core, We_core, tau_core, w_c, U_core = \
                        core_props(d_core, We_core, tau_core, U, core_droplet_velocity[n-1],  props, core_accel)

                    core_droplet_sizes.append(d_core)   #Track core droplet sizes
                    core_droplet_velocity.append(U_core)    #velocities
                    breakup_weights.append(w_c*breakup_weights[n]) #note, last one doesn't break...


        else:
            #We_c = 8.8, but also need V_c>0
            while (We_core > 8.8) and check_core_props(d_core, We_core, tau_core,  props):
                n+=1    #Count core breakup event
                all_breakup.append(single_breakup(d_core, We_core, tau_core, breakup_weights[n],  props)) #Compute core

                #Compute new core conditions
                d_core, We_core, tau_core, w_c, U_core = \
                    core_props(d_core, We_core, tau_core, U, core_droplet_velocity[n-1],  props, core_accel)

                core_droplet_sizes.append(d_core)   #Track core droplet sizes
                core_droplet_velocity.append(U_core)    #velocities
                breakup_weights.append(w_c*breakup_weights[n]) #note, last one doesn't break...


    """ Unpacking + interpreting as distribution """

    if unpack == 0:
        """
        The Model of Many modes - no lumping.
            - each breakup event gets 3 modes. No lumping of modes.
            - Remaining (unbroken) core is neglected.
        """

        # sizes = np.logspace(-1,3,1000)
        sizes = np.linspace(0, d_0*1e6, 1000)
        n_dist = np.empty(len(sizes))*0
        v_dist = np.empty(len(sizes))*0

        # w_all = 0

        for i in range(len(all_breakup)):

            for j in range(0,3): #over each mode

                w = all_breakup[i][0][j]
                mean = np.mean(all_breakup[i][1][j])
                std = np.std(all_breakup[i][1][j])

                n_dist += n_distribution(mean*1e6, std*1e6, sizes, dist_func) * w
                v_dist += v_distribution(mean*1e6, std*1e6, sizes, dist_func) * w
                # w_all += w

        #Calculate d_32 from NUMBER distribution
        d_32 = simps(sizes**3 * n_dist, x=sizes) / simps(sizes**2 * n_dist, x=sizes)


    if unpack == 1:
        """ Single-lumped mode """

        char_sizes = np.array([])
        weights = np.array([])

        w_all = 0

        for i in range(len(all_breakup)):   #over each breakup event

            w_all += np.sum(all_breakup[i][0][:3])
            for j in range(0,3): #over each mode EXCEPT bag
                char_sizes = np.append(char_sizes, all_breakup[i][1][j])
                weights = np.append(weights, all_breakup[i][0][j]*np.ones(len(all_breakup[i][1][j])) / len(all_breakup[i][1][j]))

        weighted_stats = DescrStatsW(char_sizes, weights=weights, ddof=0)
        wmean = weighted_stats.mean
        wstd = weighted_stats.std

        #Absolute
        sizes = np.logspace(-1,3,10000)
        v_dist = v_distribution(wmean*1e6, wstd*1e6, sizes)
        #w_all corrects the distribution weight for the missing volume of the final, unbroken core, given by d_core
        v_dist = v_dist * w_all

        #Calculate d_32 from NUMBER distribution
        n_dist = n_distribution(wmean*1e6, wstd*1e6, sizes)
        d_32 = simps(sizes**3 * n_dist, x=sizes) / simps(sizes**2 * n_dist, x=sizes)

        # print(round(wmean*1e6,2), round(wstd*1e6,2))


    if dist_type == 'vol':
        return sizes, v_dist, d_core, d_32

    elif dist_type == 'num':
        return sizes, n_dist, d_core, d_32










def twin_fluid_breakup_distribution(d_go, d_gi, u_g, u_l, props, dist_type = 'vol', freq = True):

    """
    Twin-fluid distribution model from Jackiw & Ashgriz (2022a)

        Inputs:
            - d_go : gas outside diameter in m
            - d_gi : gas inside diameter in m
            - u_g : gas speed in m/s
            - u_l : liquid speed in m/s
            - props : list of fluid properties, as
                    props = rho_l, rho_g, mu_l, mu_g, sigma

                        rho_l (kg/m^3)
                        rho_g (kg/m^3)
                        mu_l (Pa.s)
                        mu_g (Pa.s)
                        sigma (N/m)

        Outputs:
            - if freq == True: d_32, f, bin_edges, bin_widths, bin_centers
                    - d_32 in mu m
                    - f : frequency of bin
                    - bin edges, widths, and centers : in mu m

            - if freq == False: d_32, dist, sizes
                    - d_32 in mu m
                    - dist : distribution
                    - bin edges, widths, and centers : in mu m

        kwargs:
            - dist_type : 'vol' for volume-weighted probability density
                            or 'num' for number probability density.

            - freq : if TRUE, output frequency distribution, integrated using
                histogram bins equivalent to those of the Malvern Spraytec output.

    """


    rho_l, rho_g, mu_l, mu_g, sigma = props

    #Kelvin-Helmholtz finite-shear layer(Aliseda / Raynal)
    u_r = u_g - u_l
    b_g = (d_go - d_gi) /2
    C = 2.4
    lam_KH = 2*C*(rho_l/rho_g)**0.5*(mu_g/rho_g/u_r*b_g)**0.5


    #Liquid stream thinning

    # u_c = (np.sqrt(rho_l)*u_l + np.sqrt(rho_g)*u_g)/(np.sqrt(rho_l)+np.sqrt(rho_g))
    C = 0.1
    u_c = np.sqrt(u_l**2 + C*(rho_g/rho_l)*u_g**2)

    d_0 = d_gi * (u_l/u_c)**(1/2) #Stream thinning

    # print((u_l/u_c)**(1/2))


    #Calculate relevant non-dim parameters
    u_c = (np.sqrt(rho_l)*u_l + np.sqrt(rho_g)*u_g)/(np.sqrt(rho_l)+np.sqrt(rho_g))
    u = u_g - u_c


    sizes, dist, d_core, d_32 = breakup_distribution(d_0, u, props, \
                         Model_Core_Breakup = True, core_accel = False, \
                         dist_func = 'lognorm', dist_type = dist_type, unpack = 1)


    if freq == True:

        #Convert to volume-frequency

        #Bins from Malvern

        hist_bins = \
            np.array([1.00000200e-01, 1.16591670e-01, 1.35935900e-01, 1.58489630e-01,
                1.84785340e-01, 2.15443890e-01, 2.51189140e-01, 2.92865040e-01,
                3.41455550e-01, 3.98107950e-01, 4.64159790e-01, 5.41170600e-01,
                6.30958560e-01, 7.35643680e-01, 8.57697610e-01, 1.00000191e+00,
                1.16591668e+00, 1.35935903e+00, 1.58489633e+00, 1.84785342e+00,
                2.15443897e+00, 2.51189137e+00, 2.92865038e+00, 3.41455555e+00,
                3.98107958e+00, 4.64159775e+00, 5.41170597e+00, 6.30958605e+00,
                7.35643721e+00, 8.57697582e+00, 1.00000200e+01, 1.16591673e+01,
                1.35935907e+01, 1.58489628e+01, 1.84785347e+01, 2.15443897e+01,
                2.51189136e+01, 2.92865028e+01, 3.41455574e+01, 3.98107948e+01,
                4.64159813e+01, 5.41170578e+01, 6.30958595e+01, 7.35643692e+01,
                8.57697601e+01, 1.00000198e+02, 1.16591667e+02, 1.35935913e+02,
                1.58489624e+02, 1.84785339e+02, 2.15443893e+02, 2.51189133e+02,
                2.92865021e+02, 3.41455566e+02, 3.98107941e+02, 4.64159790e+02,
                5.41170593e+02, 6.30958557e+02, 7.35643677e+02, 8.57697571e+02,
                1.00000195e+03])



        #For use in plotting histogram using bar chart
        bin_edges = np.empty(len(hist_bins)-1)
        bin_widths = np.empty(len(hist_bins)-1)
        bin_centers = np.empty(len(hist_bins)-1)
        for i in range(len(hist_bins)-1):
            bin_edges[i] = hist_bins[i]
            bin_widths[i] = hist_bins[i+1] - hist_bins[i]
            bin_centers[i] = np.sqrt(hist_bins[i]*hist_bins[i+1])


        f = np.empty(len(bin_edges))

        for k in range(1, len(bin_edges)):

            j_1 = np.where((sizes>=bin_edges[k-1]))
            j_2 = np.where(sizes<bin_edges[k])

            j_b = np.intersect1d(j_1, j_2)

            # print(j)

            f[k] = simps(dist[j_b], x = sizes[j_b])# * (sizes[j_b[-1]] - sizes[j_b[0]])

        # print(np.sum(f))
        # print(np.sum(f*100/np.sum(f)))

        # f = f*100/np.sum(f)
        f = f[1:]*100/np.sum(f[1:])
        bin_edges = bin_edges[1:]
        bin_widths = bin_widths[1:]
        bin_centers = bin_centers[1:]


        # return d_32, all_dist, sizes
        return d_32, f, bin_edges, bin_widths, bin_centers


    else:
        return d_32, dist, sizes