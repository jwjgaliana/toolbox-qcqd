# toolbox-qcqd
Tools for theoretical chemistry (quantum chemistry and quantum dynamics).
Interfaces with QCQD softwares. 
Most of the scripts are for extracting data from outputs of QCQD softwares (Gaussian, Quantics, openMolcas, SHARC), and for further analysis of such data.
## Contents
### Modules and interface/analysis scripts
- CONSTANTS.py, intended for definition of physical/numerical constants and general python auxiliary functions. 
- TOOLBOX.py, mostly contains read/write i/o interface for Gaussian, non-exhaustively
    - read .fchk files for energy derivatives (mass-weighted or not, first and second order)
    - read .fchk files for vibrational analysis (from second order derivatives)
    - read .fchk files for numerical branching space evaluation
    - read .fchk files for reading and visualizing results of FCHT module (vibronic = electronic transition + vibrationally resolved transitions)
- openmolcas_utilities contains:
    - h5_to_E_DIP_SFS to read rassi.h5 to get d0i(real) dipoles and transition energies of spin-free (singlets, triplets...) states
    - h5_to_E_DIP_SOS to read rassi.h5 to get d0i(real,imag) dipoles and transition energies of spin-orbit coupled states 
    - h5_to_DDM_CUB_from_trd1_rassi_via_NDO to read rasscf.h5 and rassi.h5 to get transition density matrices, associated NTOs and transition density $\rho_{0i}(\mathbf{r})$
    - do_charges_SFS{,\_pop}.py to read rasscf.h5 and rassi.h5 to get density difference matrices, associated NDOs and density difference $\Delta\rho_{0i}(\mathbf{r})$
    - h5_to_TDM_CUB_from_trd1_rassi_via_NTO to read rasscf.h5 and rassi.h5 to get density difference matrices, associated NDOs and density difference $\Delta\rho_{0i}(\mathbf{r})$ 
      [computes charges and/or cube files for basins and integrated electron density in basins]
### Notebooks for future pedagogical use
- fchk2NormalModes, contains Jupyter Notebook to obtain NMV from Hessian calculations, read and visualize them
- fchk2NumericalBranchingSpace, contains Jupyter Notebook to obtained (numerical) branching space from two Hessian calculations at a CoIn geometry [Gonon et al., JCP, 2017]
- autocorrelation2spectrum, contains Jupyter Notebook to demonstrate the use of autocorrelation functions ($\langle\psi(t)|\psi(0)\rangle$) for the study of vibronic spectra (absorption and emission) beyond the Born-Oppenheimer and harmonic approximations
## TODO-list
- [ ] clean-up in TOOLBOX.py
- [ ] replace hard-coded CONSTANTS with proper use of scipy.constants
- [ ] re-organization of examples and tests
- [X] add molcas/h5 to natural transition (difference) orbitals and transition (difference) density 
- [X] add interface with Multiwfn and charge density analysis 
- [ ] add notebook for alignment of molecules and/or normal modes of vibrations [unpublished results, b234 PPE branch, [Thesis manuscript,HAL archive](https://theses.hal.science/tel-04680925/)]

## troubleshooting
- Procrustes import in TOOLBOX.py might cause problems; try commenting it if errors regarding "UserDict" are raised.
