#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys,os
from scipy import linalg
import pandas as pd

pd.set_option('display.max_rows',None)

import TOOLBOX as TLB
import CONSTANTS as CST

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    # from whichcraft import which
    from shutil import which
    return which(name) is not None

## TODO def collect_modes():
    # arguments=sys.argv[2:]
    # modes=[]
    # for argument in arguments:
        # try:
            # mode=int(argument)
            # modes.append(mode)
        # except ValueError:
            # print(f"Ignored : {argument} is not a valid integer.")
    # return modes

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=
        """
        Numerical Branching Space tool.
        Inspired from Gonon /et al./, The Journal of Chemical Physics, 2017.

        The program takes as inputs:
        - two fchk files from frequency calculation     {filetag}_{A,B}.fchk or filetag_{rootA,rootB}.fchk 
        and extracts the Hessians and gradients of energy of states A and B to compute the Hessian
        of the squared energy difference.
        At a Conical Intersection locus, the diagonalization of such a Hessian yields two non-zero 
        eigenvalues, associated to the branching space of the CoIn.

        - TODO print the BSV as if there were results of a Gaussian calculation (as in a .log)
        """
    )

    parser.add_argument("--filetag",metavar="filetag",required=True,help="Tag (no extension) of the files to be read")
    parser.add_argument("--NAtoms",metavar="NAtoms",required=True,help="Number of atoms")
    parser.add_argument("--rootA",metavar="rootA",required=True,help="Index of first excited state")
    parser.add_argument("--rootB",metavar="rootB",required=True,help="Index of second excited state")
    parser.add_argument("--roottag",metavar="roottag",required=False,help="Suffix for the states ({filetag}_{1,2} with num or {filetag}_{A,B} with AB)",default="AB")
    parser.add_argument("--molecule",metavar="molecule",required=False,default="Molecule",help="Name/tag of the molecule")
    parser.add_argument("--save",metavar="save",required=False,default="no",help="y or yes to save the figures as {filetag}_{NBS1,2}.png")
    parser.add_argument("--save_asNM",metavar="save_asNM",required=False,default="yes",help="y or yes to save the figures as {filetag}_NBSCart_x{1,2}.pdf")
    parser.add_argument("--highest_negative",metavar="highest_negative",required=False,default="yes",help="y or yes to save the figures as {filetag}_{NBS1,2}.png")
    args=vars(parser.parse_args())

    filetag=args["filetag"]
    NAtoms=int(args["NAtoms"])
    rootA=args["rootA"]
    rootB=args["rootB"]
    roottag=args["roottag"]
    save=args["save"]
    save_asNM=args["save_asNM"]
    molecule=args["molecule"]
    highest_negative=str(args["highest_negative"])

    NCoords=3*NAtoms
    NModes=3*NAtoms-6

    if roottag=="num":
        filenameA=filetag+"_"+str(rootA)+".fchk"
        filenameB=filetag+"_"+str(rootB)+".fchk"
    elif roottag=="AB":
        filenameA=filetag+"_A.fchk"
        filenameB=filetag+"_B.fchk"
    currentEnergySA,currentGradientSA,currentHessianSA=TLB.fchk2derivatives(filenameA,mw=True,freq=True)[:3]
    currentEnergySB,currentGradientSB,currentHessianSB=TLB.fchk2derivatives(filenameB,mw=True,freq=True)[:3]
    currentGradientSA=currentGradientSA.flatten() # Hartree / (Bohr·me^1/2)
    currentGradientSB=currentGradientSB.flatten() # Hartree / (Bohr·me^1/2)

    frequencies,reducedMasses,printedModesCart,atomicNumbers,atomicMasses,coordinates=TLB.fchk2vibrationalAnalysis(filenameA)
    atomicMasses=atomicMasses=TLB.fchk2derivatives(filenameA)[3]
    currentCoordinates=coordinates.flatten() # Angstrom

    currentEnergyDifference=currentEnergySB-currentEnergySA
    currentGradientDifference=currentGradientSB-currentGradientSA
    currentGradientDifferenceNorm=np.sqrt(np.sum(currentGradientDifference**2))
    currentGradientAverage=0.5*(currentGradientSA+currentGradientSB)

    currentHessianDifference=currentHessianSB-currentHessianSA
    currentGradientDifferenceProjector=np.tensordot(0.5*currentGradientDifference,0.5*currentGradientDifference,axes=0)
    currentSquaredHessianDifference=2*(0.5*currentEnergyDifference)*(0.5*currentHessianDifference)+2*currentGradientDifferenceProjector
    eigval,diagonalizer=linalg.eigh(currentSquaredHessianDifference)
    eigvec=diagonalizer.T
    BSVLengths,BSVVectors=eigval[-2:][::-1],eigvec[-2:][::-1]

    inverseSquareRootMasses=np.diag(1/np.sqrt(atomicMasses.flatten()*CST.AMU_TO_ME)) 
    BSVVectorsCart=np.dot(inverseSquareRootMasses,BSVVectors.T) # vectors(mw coords) --> vectors(cart coords)
    normalizationFactors=np.sqrt(1/np.sum(BSVVectorsCart**2,axis=0)) # sqrt(m_e)
    BSVReducedMasses=1/np.sum(BSVVectorsCart**2,axis=0)/CST.AMU_TO_ME # m_e --> AMU
    BSVVectorsCart=normalizationFactors*BSVVectorsCart # not normalized --> normalized 
    BSVVectorsCart=BSVVectorsCart.T

    if save_asNM=="y" or save_asNM=="yes":
        grid=False
        modeSelection=[0,1]
        for mode_index in modeSelection:
            fig,ax=TLB.visualizeNormalMode(coordinates.flatten(),BSVVectorsCart[mode_index],atomicNumbers=atomicNumbers,projection="2d")
            ax.set_aspect("equal")
            # fig.suptitle("{} mode {}".format(filetag,mode))
            if grid:
                plt.grid()
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            else:
                plt.axis("off")
            plt.tight_layout()
            pdfname="{}_NBSCart_x{}.pdf".format(filetag,mode_index+1)
            plt.savefig(pdfname)
            plt.show()
            if is_tool("pdfcrop"):
                os.system("pdfcrop {}".format(pdfname))


    print("NAtoms, NCoords: {}, {}".format(NAtoms,NCoords))
    print("Branching-space vectors shape: ",BSVVectors.shape)
    print("Associated Lengths: ",BSVLengths)
    print("u1·u1 = {}".format(np.dot(BSVVectors[0],BSVVectors[0]))) # normalized 
    print("u2·u2 = {}".format(np.dot(BSVVectors[1],BSVVectors[1]))) # normalized 
    print("u1·u2 = {}".format(np.dot(BSVVectors[0],BSVVectors[1]))) # and orthogonal
    print("ucart1·ucart1 = {}".format(np.dot(BSVVectorsCart[0],BSVVectorsCart[0]))) # normalized
    print("ucart2·ucart2 = {}".format(np.dot(BSVVectorsCart[1],BSVVectorsCart[1]))) # normalized
    print("ucart1·ucart2 = {}".format(np.dot(BSVVectorsCart[0],BSVVectorsCart[1]))) # but not rigorously orthogonal

    BranchingSpaceVector1=np.sqrt(BSVLengths[0]/2)*BSVVectors[0] # E_h/(a_0·sqrt(m_e))
    BranchingSpaceVector2=np.sqrt(BSVLengths[1]/2)*BSVVectors[1] # E_h/(a_0·sqrt(m_e))
    print("x1·x1 = {}".format(np.dot(BranchingSpaceVector1,BranchingSpaceVector1))) # not normalized
    print("x2·x2 = {}".format(np.dot(BranchingSpaceVector2,BranchingSpaceVector2))) # not normalized
    print("x1·x2 = {}".format(np.dot(BranchingSpaceVector1,BranchingSpaceVector2))) # and orthogonal

    #filetag="m22_opt-freq_S0.fchk"
    filenameS0="../m22_opt-freq_S0.fchk"
    frequencies,reducedMasses,printedModesCart,atomicNumbers,atomicMasses,coordinates_fcp=TLB.fchk2vibrationalAnalysis(filenameS0,gaussianType=True)
    df_vib=pd.DataFrame({"Freq (rcm)":frequencies,"Freq (a.u.)":(frequencies/CST.HARTREE_TO_RCM)**2,"μ (AMU)":reducedMasses,"μ (m_e)":reducedMasses*CST.AMU_TO_ME})
    df_vib.index=np.append(["TR"]*6,range(1,len(coordinates_fcp)+1-6))
    print(df_vib)
    print(df_vib.to_latex())

    normalModesS0=TLB.orthogonalizeNormalModes(printedModesCart,reducedMasses,atomicMasses,check=True)

    BranchingSpaceVector1_InNM=np.dot(normalModesS0,BranchingSpaceVector1)
    PercentagesNM_InBSV1=(BranchingSpaceVector1_InNM/np.sqrt(BSVLengths[0]/2))**2
    print("Sum of the percentages of NMS0 in BSV1",np.sum(PercentagesNM_InBSV1))
    BranchingSpaceVector2_InNM=np.dot(normalModesS0,BranchingSpaceVector2)
    PercentagesNM_InBSV2=(BranchingSpaceVector2_InNM/np.sqrt(BSVLengths[1]/2))**2
    print("Sum of the percentages of NMS0 in BSV2",np.sum(PercentagesNM_InBSV2))
    GradientAverage_InNM=np.dot(normalModesS0,currentGradientAverage)
    currentGradientAverageNorm=np.sqrt(np.sum(currentGradientAverage**2))
    PercentagesNM_InGA=(GradientAverage_InNM/currentGradientAverageNorm)**2
    print("Sum of the percentages of NMS0 in GA",np.sum(PercentagesNM_InGA))
    # Should be equal
    # print(np.sqrt(np.sum(BranchingSpaceVector1_InNM**2)))
    # print(np.sqrt(BSVLengths[0]/2))
    # are equal
    df=pd.DataFrame({"BSV1 (NMS0)":BranchingSpaceVector1_InNM,
                     "%NMS0 in BSV1":100*np.round(PercentagesNM_InBSV1,3),
                     "BSV2 (NMS0)":BranchingSpaceVector2_InNM,
                     "%NMS0 in BSV2":100*np.round(PercentagesNM_InBSV2,3),
                     "GA (NMS0)":GradientAverage_InNM,
                     "%NMS0 in GA":100*np.round(PercentagesNM_InGA,3)})
    df.index=np.append(["TR"]*6,range(1,len(coordinates)+1-6))
    print(df)
    print(df.to_latex())


    with pd.option_context('display.float_format', '{:0.3f}'.format):
        df=pd.DataFrame({"BSV1 (NMS0)":1000*BranchingSpaceVector1_InNM,
                     "BSV2 (NMS0)":1000*BranchingSpaceVector2_InNM,
                     "GA (NMS0)":1000*GradientAverage_InNM})
        df.index=np.append(["TR"]*6,range(1,len(coordinates)+1-6))
        print(df)
        print(df.to_latex())
    with pd.option_context('display.float_format', '{:0.1f}'.format):
        df=pd.DataFrame({"%NMS0 in BSV1":100*np.round(PercentagesNM_InBSV1,3),
                     "%NMS0 in BSV2":100*np.round(PercentagesNM_InBSV2,3),
                     "%NMS0 in GA":100*np.round(PercentagesNM_InGA,3)})
        df.index=np.append(["TR"]*6,range(1,len(coordinates)+1-6))
        print(df)
        print(df.to_latex())
