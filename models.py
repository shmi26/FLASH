#
# Sep 13, 2023: Nondimensionalization + making consistent with Shivangi
#               making De and Wi center-stage
#
# The nonlinear part of a model is the only term that needs modification
#
# Right now, additional dynamical variables cannot be easily set
# q = [sigma11, sigma22, sigma12] is hard-coded.
#
#

import numpy as np

def fnl_ode(ti, q1, q2, q3, params, model='giesekus'):
    """
    This is the interface to the nonlinear component of the constitutive model.
    
    Can be specialized for any model. Will have to tune params accordingly, or perhaps
    later add another term called model_params
    """

    De, Wi, H, N, model, theta = unfurl_parameters(params)

    lam = theta[1]
    w = De/lam; gam0 = Wi/De

    gam_dot = np.cos(De * ti)

    if model == 'giesekus':

        G, lam, alpha = theta
        const = alpha * Wi

        fnl1 = q1 + const * (q1**2 + q3**2) - 2 * Wi * gam_dot * q3 
        fnl2 = q2 + const * (q2**2 + q3**2)
        fnl3 = q3 + const * ((q1 + q2) * q3) - Wi * gam_dot * q2
        
    elif model == 'ucm':
        G, lam = theta

        fnl1 = q1 - 2 * Wi * gam_dot * q3 
        fnl2 = q2 
        fnl3 = q3 - Wi * gam_dot * q2        
        
    elif model == 'ptt':

        G, lam, epsilon = theta
        f = np.exp(epsilon * Wi * (q1 + q2))

        fnl1 = q1*f - 2 * Wi * gam_dot * q3 
        fnl2 = q2*f 
        fnl3 = q3*f - Wi * gam_dot * q2
        
    elif model == 'tnm':

        G, lam, a, b = theta
        f = np.exp(a*G*Wi*np.abs(q3))
        g = np.exp(b*G*Wi*np.abs(q3))

        fnl1 = q1*g - 2 * Wi * gam_dot * q3 + (g - f)/Wi
        fnl2 = q2*g + (g - f)/Wi
        fnl3 = q3*g - Wi * gam_dot * q2
        
    return fnl1, fnl2, fnl3


def unfurl_parameters(params):
    """
    Read parameters "params" and unfurl them into individual elements
    This is used in most functions to simplify access to params elements
    """
    # operating parameters
    De    = params['De']
    Wi    = params['Wi']

    # algorithmic parameters
    H     = params['H']
    N     = params['N']  

    # constitutive model parameters
    model = params['model']   
    theta = params['theta']   # decide order G, lam, alpha for Geisekus

    return De, Wi, H, N, model, theta
