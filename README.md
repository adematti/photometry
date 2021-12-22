# Photometry

## Introduction

This package gathers Python routines to analyse variations of target density with photometric properties of
the [Legacy Survey](https://www.legacysurvey.org/). Especially, it includes:
- `target_selection.py`: a convenient (more than numpy structured array) class to handle data catalogues
- `models.py`: so far, linear template regression model
- `MCtool.py`: tool to forward model the impact of photometric errors onto target density, can run in parallel with MPI
- `density_variations.py`: classes to plot variations of the target density with respect to photometric templates, with HEALPix, bricks, and direct data/randoms comparison
- `correlation_function.py`: classes to compute and plot the angular correlation function and power spectrum
Documentation can be provided on request.

## Tests

All tests are in tests/. NERSC notebooks are in NERSC/.
They can be run straightforwardly on [JupyterHub](https://jupyter.nersc.gov/hub/home).

## License

See the [LICENSE](https://github.com/adematti/obiwan/blob/master/LICENSE).

## Requirements

- Python 3
- numpy
- scipy
- matplotlib
- fitsio
- mpi4py
- healpy
