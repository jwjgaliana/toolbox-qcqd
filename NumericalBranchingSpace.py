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
        - a geometry                                    {filetag}.xyz
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

    with open(filetag+"_NBS.dat","w") as ResultsFile:
        ResultsFile.write("Numerical Branching Space information for "+filetag+" calculations")
        ResultsFile.write("\n")
        ResultsFile.write("\n")

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

    currentHessianDifference=currentHessianSB-currentHessianSA
    currentGradientDifferenceProjector=np.tensordot(0.5*currentGradientDifference,0.5*currentGradientDifference,axes=0)
    currentSquaredHessianDifference=2*(0.5*currentEnergyDifference)*(0.5*currentHessianDifference)+2*currentGradientDifferenceProjector
    eigval,diagonalizer=linalg.eigh(currentSquaredHessianDifference)
    eigvec=diagonalizer.T
    BSVLengths,BSVVectors=eigval[-2:][::-1],eigvec[-2:][::-1]

    with open(filetag+"_NBS.dat","a+") as ResultsFile:
        ResultsFile.write("Eigenvalues of the Hessian of the squared energy difference:")
        ResultsFile.write("\n")
        ResultsFile.write("\n")
        for i in range(len(eigval)):
            ResultsFile.write(str(i+1)+"\t"+str(eigval[i]))
            ResultsFile.write("\n")
        ResultsFile.write("\n")

        ResultsFile.write("NAtoms, NCoords: {}, {}\n".format(NAtoms,NCoords))
        ResultsFile.write("Branching-space vectors shape: \n".format(BSVVectors.shape))
        ResultsFile.write("Associated Lengths: \n".format(BSVLengths))
        ResultsFile.write("u1·u1 = {}\n".format(np.dot(BSVVectors[0],BSVVectors[0]))) # normalized
        ResultsFile.write("u2·u2 = {}\n".format(np.dot(BSVVectors[1],BSVVectors[1]))) # normalized
        ResultsFile.write("u1·u2 = {}\n".format(np.dot(BSVVectors[0],BSVVectors[1]))) # and orthogonal

    currentHighestNegative=eigvec[0].flatten()
    currentBranchingSpaceVector1=eigvec[-1].flatten()
    currentBranchingSpaceVector2=eigvec[-2].flatten()

    current_coordinates=currentCoordinates.reshape(NAtoms,3)
    current_HighestNegative=currentHighestNegative.reshape(NAtoms,3) 
    current_BranchingSpaceVector1=currentBranchingSpaceVector1.reshape(NAtoms,3) 
    current_BranchingSpaceVector2=currentBranchingSpaceVector2.reshape(NAtoms,3) 

    with open(filetag+"_NBS.dat","a+") as ResultsFile:
        ResultsFile.write("Length (E_h^2/(a_0^2·m_e)) and components of the Unitary BS vector (as NM) associated to the highest non-zero eigenvalue:")
        ResultsFile.write("\n")
        ResultsFile.write("\n")
        ResultsFile.write("\t"+str(eigval[-1]))
        ResultsFile.write("\n")
        ResultsFile.write("\n")
        # ResultsFile.write("Num. Atom.\t x \t y \t z \t (in Hartree/(Angstrom * sqrt(me))")
        ResultsFile.write("Num. Atom.\t x \t y \t z")
        ResultsFile.write("\n")
        for i in range(len(current_BranchingSpaceVector1)):
            ResultsFile.write(str(i+1)+"\t\t"+str(current_BranchingSpaceVector1[i][0])+"\t\t"+str(current_BranchingSpaceVector1[i][1])+"\t\t"+str(current_BranchingSpaceVector1[i][2])+"\n")
        ResultsFile.write("\n")
        ResultsFile.write("Length (E_h^2/(a_0^2·m_e)) and components of the Unitary BS vector (as NM) associated to the lowest non-zero eigenvalue:")
        ResultsFile.write("\n")
        ResultsFile.write("\n")
        ResultsFile.write("\t"+str(eigval[-2]))
        ResultsFile.write("\n")
        ResultsFile.write("\n")
        # ResultsFile.write("Num. Atom.\t x \t y \t z \t (in Hartree/(Angstrom * sqrt(me))")
        ResultsFile.write("Num. Atom.\t x \t y \t z")
        ResultsFile.write("\n")
        for i in range(len(current_BranchingSpaceVector2)):
            ResultsFile.write(str(i+1)+"\t\t"+str(current_BranchingSpaceVector2[i][0])+"\t\t"+str(current_BranchingSpaceVector2[i][1])+"\t\t"+str(current_BranchingSpaceVector2[i][2])+"\n")


    inverseSquareRootMasses=np.diag(1/np.sqrt(atomicMasses.flatten()*CST.AMU_TO_ME)) 
    BSVVectorsCart=np.dot(inverseSquareRootMasses,BSVVectors.T) # vectors(mw coords) --> vectors(cart coords)
    normalizationFactors=np.sqrt(1/np.sum(BSVVectorsCart**2,axis=0)) # sqrt(m_e)
    BSVReducedMasses=1/np.sum(BSVVectorsCart**2,axis=0)/CST.AMU_TO_ME # m_e --> AMU
    BSVVectorsCart=normalizationFactors*BSVVectorsCart # not normalized --> normalized 
    BSVVectorsCart=BSVVectorsCart.T

    with open(filetag+"_NBS.dat","a+") as ResultsFile:
        ResultsFile.write("Normalization Factor (a.u.):\n {} \n".format(normalizationFactors))
        ResultsFile.write("reducedMasses (AMU):\n {} \n".format(BSVReducedMasses))
        ResultsFile.write("ucart1·ucart1 = {}\n".format(np.dot(BSVVectorsCart[0],BSVVectorsCart[0]))) # normalized
        ResultsFile.write("ucart2·ucart2 = {}\n".format(np.dot(BSVVectorsCart[1],BSVVectorsCart[1]))) # normalized
        ResultsFile.write("ucart1·ucart2 = {}\n".format(np.dot(BSVVectorsCart[0],BSVVectorsCart[1]))) # but not rigorously orthogonal


    fig=plt.figure()
    ax=fig.add_subplot(111,projection="3d")
    ax.scatter(current_coordinates[:,0],current_coordinates[:,1],current_coordinates[:,2])
    ax.scatter(current_coordinates[:,0]+current_BranchingSpaceVector1[:,0],current_coordinates[:,1]+current_BranchingSpaceVector1[:,1],current_coordinates[:,2]+current_BranchingSpaceVector1[:,2])
    max_coordinates=np.max(np.abs(current_coordinates))
    ax.set_title("NBS 1 with ||NBS₁||="+str(eigval[-1]))
    ax.set_xlim(-max_coordinates,max_coordinates)
    ax.set_ylim(-max_coordinates,max_coordinates)
    ax.set_zlim(-max_coordinates,max_coordinates)
    if save=="y" or save=="yes":
        plt.savefig(filetag+"_NBS1.png")
        plt.savefig(filetag+"_NBS1.pdf")
    plt.show()
    fig=plt.figure()
    ax=fig.add_subplot(111,projection="3d")
    ax.scatter(current_coordinates[:,0],current_coordinates[:,1],current_coordinates[:,2])
    ax.scatter(current_coordinates[:,0]+current_BranchingSpaceVector2[:,0],current_coordinates[:,1]+current_BranchingSpaceVector2[:,1],current_coordinates[:,2]+current_BranchingSpaceVector2[:,2])
    ax.set_title("NBS 2 with ||NBS₂||="+str(eigval[-2]))
    ax.set_xlim(-max_coordinates,max_coordinates)
    ax.set_ylim(-max_coordinates,max_coordinates)
    ax.set_zlim(-max_coordinates,max_coordinates)
    if save=="y" or save=="yes":
        plt.savefig(filetag+"_NBS2.png")
        plt.savefig(filetag+"_NBS2.pdf")
    plt.show()
    if highest_negative=="y" or highest_negative=="yes":
        fig=plt.figure()
        ax=fig.add_subplot(111,projection="3d")
        ax.scatter(current_coordinates[:,0],current_coordinates[:,1],current_coordinates[:,2])
        ax.scatter(current_coordinates[:,0]+current_HighestNegative[:,0],current_coordinates[:,1]+current_HighestNegative[:,1],current_coordinates[:,2]+current_HighestNegative[:,2])
        ax.set_title("eigvec 0 with eigval="+str(eigval[0]))
        ax.set_xlim(-max_coordinates,max_coordinates)
        ax.set_ylim(-max_coordinates,max_coordinates)
        ax.set_zlim(-max_coordinates,max_coordinates)
    plt.show()


    print("Save as NM:",save_asNM)
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
