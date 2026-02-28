# Computational-Determination-of-Solar-Rotation-Periods-via-Sunspot-Tracking
This repository contains the Python code to track the sunspot over a period of time and find how fast the sun rotates around its axis.

How it works:
1. **Data Extraction:** Download daily HMI Intensitygrams using Python (in FITS format).
2. **Coordinate Transformation:** Read the exact FITS metadata ($x_0,y_0,R$) to map the 2D coordinates into 3D solar lattitudes and longitudes.
3. **Some Corrections to take into account:** Convert the synodic period of the Sun into the real Sidereal period.
4. **Curve Fitting:** Calculate the differential rotation coefficients ($A$ and $B$).

## Softwares and libraries used:

* Python 3
* Numpy
* Astropy
* Scipy
* Sunpy
* Pandas
* Matplotlib

