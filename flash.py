#
# FLASH CODE
# date: Feb 1, 2024
#
# To run, modify the input file inp.dat (model, De, Wi, and #harmonics)
# The code saves the nondimensional FCs sigma1_hat, sigma22_hat and sigma12_hat as q1hat.dat, q2hat.dat, and q3hat.dat, respectively.
# 
# skeletal FLASH code for inclusion as Supplementary Material with Physics of Fluids paper
# 
# One difference from the text is that sigma33 is assumed so that
#       q = [q1 = sigma11, q2=sigma22, q3=sigma12]
#
#

import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

plt.style.use('bmh')		

from matplotlib import rcParams
rcParams['axes.labelsize'] = 20 
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14 
rcParams['legend.fontsize'] = 16
rcParams['lines.linewidth'] = 3

import fourier as F
from models import fnl_ode, unfurl_parameters

def analUCMSolution(params):
    """
    analytical solution for LVE response of UpperConvectedMaxwell model from Ferry. 
    For ease of comparison, the analytical solution has a shape determined by H.
    """
    # unfurl params
    De, Wi, H, N, model, theta = unfurl_parameters(params)
    G, lam = theta[0], theta[1]

    gam0 = Wi/De

    # moduli G*(w) and G*(2w). The latter shows up in tau111
    Gp = G*De**2/(1 + De**2)
    Gpp = G*De/(1 + De**2)

    Gp2  = G*(2*De)**2/(1 + (2*De)**2)
    Gpp2 = G*(2*De)/(1 + (2*De)**2)

    # furnish vectors of compliant shape
    q1hat_anal = np.zeros(2*H+1)
    q2hat_anal = np.zeros(2*H+1)
    # ~ q3hat_anal = np.zeros(2*H+1)
    # ~ q4hat_anal = np.zeros(2*H+2)
    q3hat_anal = np.zeros(2*H+2)

    # populate only the nonzero terms
    # ~ q4hat_anal[0]   = gam0 * Gpp
    # ~ q4hat_anal[H+1] = gam0 * Gp

    q3hat_anal[0]   = gam0 * Gpp
    q3hat_anal[H+1] = gam0 * Gp

        
    q1hat_anal[0] = 2 * gam0**2 * Gp
    q1hat_anal[1] = gam0**2 * (-Gp + 0.5*Gp2)
    q1hat_anal[H+1] = gam0**2 * (Gpp - 0.5*Gpp2)
    
    # ~ qhat_anal = np.concatenate([q1hat_anal, q2hat_anal, q3hat_anal, q4hat_anal])
    # normalized by G*Wi
    qhat_anal = np.concatenate([q1hat_anal, q2hat_anal, q3hat_anal])/(G*Wi)

    return qhat_anal
    
def FC_qdot(qhat, params):
    """
    Return Fourier Coefficients (FC) of qdot, given the FC qhat of q(t)
    The returned vector of FCs has the same shape as qhat
    """

    De, Wi, H, N, model, theta = unfurl_parameters(params)

    qdot_hat = np.zeros(len(qhat))
    
    # relevant harmonics
    l1  = np.arange(2, 2*H+1, 2)
    l3  = np.arange(1, 2*H+2, 2)

    # q1
    qdot_hat[0] = 0.
    qdot_hat[1:H+1] = De * l1 * qhat[H+1:2*H+1] 
    qdot_hat[H+1:2*H+1] = -De * l1 * qhat[1:H+1] 
    
    # q2
    qdot_hat[2*H+1] = 0.
    qdot_hat[2*H+2:3*H+2] =  De * l1 * qhat[3*H+2:4*H+2]
    qdot_hat[3*H+2:4*H+2] = -De * l1 * qhat[2*H+2:3*H+2]

    # new q3
    qdot_hat[4*H+2:5*H+3] =   De * l3 * qhat[5*H+3:6*H+4]
    qdot_hat[5*H+3:6*H+4] =  -De * l3 * qhat[4*H+2:5*H+3]    

    return qdot_hat

def qt_compute(qhat, params, isPlot=False):
    """
    for AFT: reconstruct q1(t), q2(t), q3(t) samples at N sampling points from the FCs
    """
    De, Wi, H, N, model, theta = unfurl_parameters(params)

    if isPlot: N = 2**8

    # set time-grid
    T    = 2*np.pi/De   # complicated in Shivangi code: 
    ti = np.linspace(0, T, N, endpoint=False)
    
    # extract appropriate FCs
    q1_hat = qhat[:2*H+1]
    q2_hat = qhat[2*H+1:4*H+2]
    q3_hat = qhat[4*H+2:]

    # take N samples of q1, q2, q3, q4
    q1_s = F.FS_reconstruct(ti, q1_hat, De, harmonics='even', iscomplex=False)
    q2_s = F.FS_reconstruct(ti, q2_hat, De, harmonics='even', iscomplex=False)
    q3_s = F.FS_reconstruct(ti, q3_hat, De, harmonics='odd', iscomplex=False)

    return ti, q1_s, q2_s, q3_s

def fnl_hat_compute(ti, fnl1, fnl2, fnl3, params):
    """
    Given time-domain fnl gets the appropriate FC respecting the harmonics
    """
    De, Wi, H, N, model, theta = unfurl_parameters(params)
        
    fnl1_hat = F.FC_extract(ti, fnl1, H, harmonics='even', iscomplex = False)
    fnl2_hat = F.FC_extract(ti, fnl2, H, harmonics='even', iscomplex = False)
    fnl3_hat = F.FC_extract(ti, fnl3, H, harmonics='odd', iscomplex = False)

    fnl_hat = np.concatenate([fnl1_hat, fnl2_hat, fnl3_hat])

    return fnl_hat

def FC_residual(qhat, params):
    """
    Given FC of guessed solution qhat, and params, returns residual of FC
    implements AFT to get fnl_hat
    rhat = q_dot_hat + fnl_hat - fex_hat
    """
    De, Wi, H, N, model, theta = unfurl_parameters(params)

    G = theta[0]
    
    # subsume the 1.0/lam*qhat term into the fnl_hat term
    q_dot_hat = FC_qdot(qhat, params)           # build FC(qdot) from FC(q)

    # Build hat_fex U*1 vector corresponding to the external forcing term
    fex_hat = np.zeros(len(qhat))
    # ~ fex_hat[6*H+3] = G*gam0*w
    # fex_hat[4*H+2] = G*gam0*w
    fex_hat[4*H+2] = 1.0   # tilde{dot{gamma}} = sin tilde{w} tilde{t} = 1 in right spot 

    # nonlinear part: AFT algorithm
    # (1) get time domain q and fnl
    # (2) convert to Fourier domain
    # ~ ti, q1_s, q2_s, q3_s, q4_s = qt_compute(qhat, params)                         # step 1
    # ~ fnl1_s, fnl2_s, fnl3_s, fnl4_s = fnl_ode(ti, q1_s, q2_s, q3_s, q4_s, params)  # step 2
    # ~ fnl_hat = fnl_hat_compute(ti, fnl1_s, fnl2_s, fnl3_s, fnl4_s, params)         # step 3

    ti, q1_s, q2_s, q3_s   = qt_compute(qhat, params)
    fnl1_s, fnl2_s, fnl3_s = fnl_ode(ti, q1_s, q2_s, q3_s, params)  # step 1
    fnl_hat = fnl_hat_compute(ti, fnl1_s, fnl2_s, fnl3_s, params)   # step 2

    r_hat = q_dot_hat + fnl_hat - fex_hat

    return r_hat

def plotInitFin(qold_hat, qnew_hat, params):
    """
    helper plotting function (used by HB_solve) to plot initial guess and final solution
    as function of time 
    """    
    ti, q1, q2, q3= qt_compute(qold_hat, params, isPlot=True)
    
    ti = ti/(2*np.pi) * params['De']
    plt.plot(ti, q1, '--', alpha=0.5)
    plt.plot(ti, q2, '--', alpha=0.5)
    plt.plot(ti, q3, '--', alpha=0.5)
    
    ti, q1, q2, q3 = qt_compute(qnew_hat, params, isPlot=True)

    ti = ti/(2*np.pi) * params['De']
    plt.plot(ti, q1, c='C0', label='11')
    plt.plot(ti, q2, c='C1', label='22')
    plt.plot(ti, q3, c='C2', label='12')
    
    plt.xlim(0, max(ti))
    plt.xlabel(r'$\omega t/2 \pi$')

    plt.ylabel(r'$\tilde{\sigma}$')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plotHarmonics(qhat, params):
    """
    helper plotting function (used by HB_solve) to plot harmonics
    """
    De, Wi, H, N, model, theta = unfurl_parameters(params)
    
    q1_hat = qhat[:2*H+1]
    q2_hat = qhat[2*H+1:4*H+2]
    q3_hat = qhat[4*H+2:]

    l      = np.arange(0, 2*H+2)
    l1     = np.arange(0, 2*H+1, 2)
    l3     = np.arange(1, 2*H+2, 2)

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8,4))


    # sigma11
    qC = np.zeros(len(l)); qS   = np.zeros(len(l))
    a0 = np.array([q1_hat[0]])
    a  = np.concatenate([a0/2., q1_hat[1:H+1]])
    b  = np.concatenate([[0.],  q1_hat[H+1:]])

    qC[l1] = np.abs(a); qS[l1] = np.abs(b)

    ax[0].bar(l-0.05, qC, width=0.1, align='center')
    ax[0].bar(l+0.05, qS, width=0.1, align='center')
    ax[0].set_ylabel(r'$\hat{\sigma}_{11}$')
    
    ylim = ax[0].get_ylim()

    # sigma22
    qC = np.zeros(len(l)); qS   = np.zeros(len(l))
    a0 = np.array([q2_hat[0]])
    a  = np.concatenate([a0/2., q2_hat[1:H+1]])
    b  = np.concatenate([[0.],  q2_hat[H+1:]])

    qC[l1] = np.abs(a); qS[l1] = np.abs(b)

    ax[1].bar(l-0.05, qC, width=0.1, align='center')
    ax[1].bar(l+0.05, qS, width=0.1, align='center')
    ax[1].set_ylabel(r'$\hat{\sigma}_{22}$')
    ax[1].set_ylim(ylim)

    qC = np.zeros(len(l)); qS   = np.zeros(len(l))
    a  = q3_hat[:H+1]
    b  = q3_hat[H+1:]

    qC[l3] = np.abs(a); qS[l3] = np.abs(b)

    ax[2].bar(l-0.05, qC, width=0.1, align='center', label='|cos|')
    ax[2].bar(l+0.05, qS, width=0.1, align='center', label='|sin|')
    ax[2].set_ylabel(r'$\hat{\sigma}_{12}$')
    ax[2].legend()

    ax[2].set_xlabel('harmonics')

    plt.tight_layout()
    plt.show()

def HB_solve(params, isPlot=False, verbose=False, useLadder=False, nLadder=4, gam0Min=1.0):
    """
    - everything is prescribed through params

    uses:
        - analUCMSolution to seed the initial solution
        - FC_residual to compute the residual for HB equation in Fourier space [2 norm]
                      and zero it find solution vector; it needs most of the other stuff
    """

    from copy import deepcopy

    De, Wi, H, N, model, theta = unfurl_parameters(params)

    # initial guess is analytical solution to UCM eqn
    # for large gam0, it is advisable to stage initialization

    if verbose:
        print("Harmonic Balance Calc.")

    t      = time()

    # if gam < 2.0, then you don't need a ladder; override
    gam0 = Wi/De
    if gam0 > gam0Min and useLadder:

        params_tmp = deepcopy(params)

        # if ladder is too sparse then densify
        minLadder = 2 + int(np.round(np.log10(gam0)))
        if nLadder < minLadder: nLadder = minLadder

        gam0v = np.geomspace(gam0Min, gam0, nLadder)

        if verbose:
            print("gam0Ladder", gam0v)

        # Do all but the actual gam0
        for i, Wi_tmp in enumerate(gam0v[:-1]*De):

            params_tmp['Wi'] = Wi_tmp
    
            if i == 0:
                qhat = analUCMSolution(params_tmp)

            qhat = fsolve(FC_residual, x0=qhat, args=(params_tmp,))

            if verbose:
                print("ladder{:d} res\t{:.4e}".format(i+1, np.linalg.norm(FC_residual(qhat, params_tmp))))

        qhat_0 = qhat

    else:
        # initial guess is analytical solution to UCM eqn
        qhat_0 = analUCMSolution(params)   
        if verbose:
            rhat = FC_residual(qhat_0, params)
            error_freq = np.linalg.norm(rhat)/np.sqrt(len(rhat))
            print("ini residual\t{:.4e}".format(error_freq))

    # solve
    qhat = fsolve(FC_residual, x0=qhat_0, args=(params,))
    telapsed = time() - t

    if verbose:
        rhat = FC_residual(qhat, params)
        error_freq = np.linalg.norm(rhat)/np.sqrt(len(rhat))
        print("fin residual\t{:.4e}".format(error_freq))
        print("time elapsed\t{:.4e}".format(telapsed))

        np.savetxt('q1hat.dat', qhat[:2*H+1])
        np.savetxt('q2hat.dat', qhat[2*H+1:4*H+2])
        np.savetxt('q3hat.dat', qhat[4*H+2:])

    t, q1, q2, q3 = qt_compute(qhat, params, isPlot=True)
    q = np.c_[q1, q2, q3]
    
    if isPlot:
        plotInitFin(qhat_0, qhat, params)
        plotHarmonics(qhat, params)

    return t, q, qhat, telapsed

def readInput(fname='inp.dat'):
    """Reads data from the input file (default = 'inp.dat')
    and populates the parameter dictionary par"""

    par  = {}

    # read the input file
    for line in open(fname):

        li=line.strip()

        # if not empty or comment line; currently list or float
        if len(li) > 0 and not li.startswith("#" or " "):

            li = line.rstrip('\n').split(':')
            key = li[0].strip()
            tmp = li[1].strip()

            val = eval(tmp)

            par[key] = val

    n = int(np.ceil(np.log2(4*par['H']+4)))
    par['N'] = 2**n

    return par

#---------------
# Main Program
#---------------
if __name__ == '__main__':
    params = readInput()
    t, q, qhat, telapse = HB_solve(params, isPlot=params['isPlot'], verbose=params['verbose'], useLadder=True)
