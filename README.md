# toolbox-qcqd
Tools for theoretical chemistry (quantum chemistry and quantum dynamics).
Interfaces with QCQD softwares. 
Most of the scripts are for extracting data from outputs of QCQD softwares, and for further analysis of such data.

## Contents
- CONSTANTS.py, intended for definition of physical/numerical constants and general python auxiliary functions. 
- TOOLBOX.py, intended for group
- autocorrelation2spectrum contains Jupyter Notebook for demonstration of the use of autocorrelation functions ($\langle\psi(t)|\psi(0)\rangle$) for the study of vibronic spectra (absorption and emission) beyond the Born-Oppenheimer and harmonic approximations
- openmolcas_utilities
    - h5_to_E_DIP_SFS read rassi.h5 to get d0i(real) dipoles and transition energies of spin-free (singlets, triplets...) states
    - h5_to_E_DIP_SOS read rassi.h5 to get d0i(real,imag) dipoles and transition energies of spin-orbit coupled states 
    - h5_to_DDM_CUB_from_trd1_rassi_via_NDO read rasscf.h5 and rassi.h5 to get transition density matrices, associated NTOs and transition density $\rho_{0i}(\mathbf{r})$
    - h5_to_TDM_CUB_from_trd1_rassi_via_NTO read rasscf.h5 and rassi.h5 to get density difference matrices, associated NDOs and density difference $\Delta\rho_{0i}(\mathbf{r})$

## TODO-list
- [ ] clean-up in TOOLBOX.py
- [ ] re-organization of examples and tests
- [X] add molcas/h5 to natural transition (difference) orbitals and transition (difference) density 
- [ ] add interface with Multiwfn and charge density analysis

## troubleshooting
- Procrustes import in TOOLBOX.py might cause problems; try commenting it if errors regarding "UserDict" are raised.
