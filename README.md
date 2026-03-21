# toolbox-qcqd
Tools for theoretical chemistry (quantum chemistry and quantum dynamics).
Interfaces with QCQD softwares. 
Most of the scripts are for extracting data from outputs of QCQD softwares, and for further analysis of such data.

## Contents
- CONSTANTS.py, intended for definition of physical/numerical constants and general python auxiliary functions. 
- TOOLBOX.py, intended for group
- autocorrelation2spectrum contains Jupyter Notebook for demonstration of the use of autocorrelation functions (<ψ(t)|ψ(0)>) for the study of vibronic spectra (absorption and emission) beyond the Born-Oppenheimer and harmonic approximations

## TODO-list
- [ ] clean-up in TOOLBOX.py
- [ ] re-organization of examples and tests
- [ ] add molcas/h5 to natural transition (difference) orbitals and transition (difference) density 
- [ ] add interface with Multwfn and charge density analysis

## troubleshooting
- Procrustes import in TOOLBOX.py might cause problems; try commenting it if errors regarding "UserDict" are raised.
