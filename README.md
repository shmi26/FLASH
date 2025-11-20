# FLASH

`flash.py` is a python program to compute the periodic steady state stress response of a nonlinear constitutive model (CM) subjected to oscillatory shear.

It implements harmonic balance (HB) with the alternating frequency time (AFT) numerical scheme as described in the paper "Harmonic Balance for Differential Constitutive Models under Oscillatory Shear" submitted to Physics of Fluids.


## Running a Calculation

### Input File

To run a calculation modify the input (text) file `inp.dat` by specifying the number of harmonics (H), the strain amplitude and frequency (via dimensionless numbers), and model details.

Currently, only the models discussed in the paper are implemented. Therefore `model` should be one of the following: 'giesekus', 'ucm', 'ptt', 'tnm'

Model parameters are specified via `theta`. The first two elements are the linear visoelastic parameters $G$ and $\lambda$, followed by the nonlinear parameter(s).

`isPlot` can be set to `True` or `False` depending on whether onscreen plots are desired.

### Calculation and Output

Run `flash.py` using
> python3 flash.py

The code saves the nondimensional Fourier coefficients corresponding to $\hat{\sigma}_{11}$, $\hat{\sigma}_{22}$, and $\hat{\sigma}_{12}$ as `q1hat.dat`, `q2hat.dat`, and `q3hat.dat`, respectively.

For the normal stresses, there are 2H+1 Fourier coefficients corresponding to $$[a_0, a_2, a_4, \cdots, a_{2H}, b_2, b_4, \cdots, b_{2H}]$$

For the shear stress, there are 2H+2 Fourier coefficients corresponding to $$[a_1, a_3, \cdots, a_{2H+1}, b_{1}, b_{3}, \cdots, b_{2H+1}]$$

The total number of unknowns that FLASH solves for is $U = 6H+4$.

## Code Organization

`flash.py` imports two files `models.py` and `fourier.py`

`models.py` contains the nonlinear differential model specifications.

`fourier.py` contains the functions for working with the Fourier series.

## Requirements
`python` 3.8
`numpy`  1.24
`scipy`  1.10
`matplotlib` 3.7



