import numpy as np

def FC_extract(t, ft, H, harmonics='even', iscomplex = False):
    """
    Extract Fourier Coefficients 

    Given t, f(t) (N samples), and number and nature of harmonics.

   
    For numerical calculations use sin-cos
    If harmonics are even, assumes fhat = [a0, a2, a4, ..., a2H, b2, b4, ..., b2H] ... 2H+1 elements
                      odd, assumes fhat = [a1, a3, ..., a2H+1, b1, b3, ..., b2H+1] ... 2H+2 elements

    Note highest harmonic = 2H for even, and 2H+1 for odd
                      
    For complex notation, only non-negative harmonics are stored.
        
    """

    fhat = np.fft.rfft(ft)/ft.size
    
    
    # truncate to highest harmonic 2H or 2H+1 and select only even or odd
    if harmonics == 'even':
        fhat = fhat[0:2*H+1:2]
        
        # if sine-cosine
        if not iscomplex:
            fhat *= 2
            a0    = fhat[0].real
            a     = fhat[1:].real
            b     = -fhat[1:].imag

            # stack all cos harmonics together etc.
            fhat = np.concatenate([np.array([a0]), a, b])
    
    else:

        fhat = fhat[1:2*H+2:2]

        if not iscomplex:
            fhat *= 2
            a     = fhat[0:].real
            b     = -fhat[0:].imag

            # concatenate harmonics
            fhat = np.empty((a.size + b.size ,), dtype=a.dtype)
            fhat = np.concatenate([a, b])
            
    return fhat

def FS_reconstruct(t, fhat, omega, harmonics='even', iscomplex=False):
    """
    Given t (N samples), Fourier coeffs fhat, and frequency omega
    reconstruct the time-domain signal
    
    Need to supply qualitative information on harmonics [even or odd]
    and whether sin-cos or exponential representation is desired.
    
    For numerical calculations use sin-cos
    If harmonics are even, assumes fhat = [a0, a2, a4, ..., a2H, b2, b4, ..., b2H] ... 2H+1 terms
                      odd, assumes fhat = [a1, a3, ..., a2H+1, b1, b3, ..., b2H+1] ... 2H+2 terms
                      
    For complex notation, only non-negative harmonics are stored.
                      
    """
    if harmonics == 'even':
        if iscomplex:
            f = fhat[0] * np.ones(len(t))
            for k in range(1, len(fhat)):
                f += fhat[k] * np.exp(2j * k * omega * t)
                f += np.conjugate(fhat[k]) * np.exp(-2j * k * omega * t)
            f = f.real
        else:
            f = 0.5 * fhat[0] * np.ones(len(t))
            
            H = int((len(fhat)-1)/2)
            
            for k in range(1, 1+H):
                f += fhat[k]   * np.cos(2*k * omega * t)
                f += fhat[k+H] * np.sin(2*k * omega * t)

    else: # harmonics = 'odd'
        if iscomplex:
            f = np.zeros(len(t), dtype=complex)
            for k in range(0, len(fhat)):
                f += fhat[k] * np.exp(1j * (2*k+1) * omega * t)
                f += np.conjugate(fhat[k]) * np.exp(-1j * (2*k+1) * omega * t)
            f = f.real
            
        else:
            f = np.zeros(len(t))
            H = int(len(fhat)/2)
            for k in range(0, H):
                f += fhat[k] * np.cos((2*k+1) * omega * t)
                f += fhat[k+H] * np.sin((2*k+1) * omega * t)
            
    return f
